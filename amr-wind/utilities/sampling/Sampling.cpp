#include <memory>
#include <utility>

#include "amr-wind/utilities/sampling/Sampling.H"
#include "amr-wind/utilities/io_utils.H"
#include "amr-wind/utilities/ncutils/nc_interface.H"

#include "AMReX_ParmParse.H"

namespace amr_wind::sampling {

Sampling::Sampling(CFDSim& sim, std::string label)
    : m_sim(sim), m_label(std::move(label))
{}

Sampling::~Sampling() = default;

void Sampling::initialize()
{
    BL_PROFILE("amr-wind::Sampling::initialize");

    // Labels for the different sampler types
    amrex::Vector<std::string> labels;
    // Fields to be sampled - requested by user
    amrex::Vector<std::string> field_names;

    {
        amrex::ParmParse pp(m_label);
        pp.getarr("labels", labels);
        pp.getarr("fields", field_names);
        pp.query("output_frequency", m_out_freq);
        pp.query("output_format", m_out_fmt);
        pp.query("output_delay", m_out_delay);
        pp.query("restart_sample", m_restart_sample);
    }

    // Process field information
    m_ncomp = 0;
    auto& repo = m_sim.repo();
    for (const auto& fname : field_names) {
        if (!repo.field_exists(fname)) {
            amrex::Print()
                << "WARNING: Sampling: Non-existent field requested: " << fname
                << std::endl;
            continue;
        }

        auto& fld = repo.get_field(fname);
        m_ncomp += fld.num_comp();
        m_fields.emplace_back(&fld);
        ioutils::add_var_names(m_var_names, fld.name(), fld.num_comp());
    }

    // Load different probe types, default probe type is line
    int idx = 0;
    m_total_particles = 0;
    for (const auto& lbl : labels) {
        const std::string key = m_label + "." + lbl;
        amrex::ParmParse pp1(key);
        std::string stype = "LineSampler";

        pp1.query("type", stype);
        auto obj = SamplerBase::create(stype, m_sim);
        obj->label() = lbl;
        obj->sampletype() = stype;
        obj->id() = idx++;
        obj->initialize(key);

        m_total_particles += obj->num_points();
        m_samplers.emplace_back(std::move(obj));
    }

    update_container();

    if (m_out_fmt == "netcdf") {
        prepare_netcdf_file();
    }

    m_sample_buf.assign(m_total_particles * m_var_names.size(), 0.0);

    if (m_restart_sample) {
        sampling_workflow();
        sampling_post();
    }
}

void Sampling::update_container()
{
    BL_PROFILE("amr-wind::Sampling::update_container");

    // Initialize the particle container based on user inputs
    m_scontainer = std::make_unique<SamplingContainer>(m_sim.mesh());

    m_scontainer->setup_container(m_ncomp);

    m_scontainer->initialize_particles(m_samplers);

    // Redistribute particles to appropriate boxes/MPI ranks
    m_scontainer->Redistribute();

    m_scontainer->num_sampling_particles() =
        static_cast<int>(m_total_particles);
}

void Sampling::update_sampling_locations()
{
    BL_PROFILE("amr-wind::Sampling::update_sampling_locations");

    for (const auto& obj : m_samplers) {
        obj->update_sampling_locations();
    }

    update_container();
}

void Sampling::post_advance_work()
{

    BL_PROFILE("amr-wind::Sampling::post_advance_work");
    const auto& time = m_sim.time();
    const int tidx = time.time_index();

    // Skip processing if delay has not been reached
    if (tidx < m_out_delay) {
        return;
    }

    // Skip processing if it is not an output timestep
    if (!(tidx % m_out_freq == 0)) {
        return;
    }

    sampling_workflow();

    process_output();

    sampling_post();
}

void Sampling::sampling_workflow()
{

    BL_PROFILE("amr-wind::Sampling::sampling_workflow");

    update_sampling_locations();

    m_scontainer->interpolate_fields(m_fields);

    fill_buffer();

    convert_velocity_lineofsight();

    create_output_buffer();
}

void Sampling::sampling_post()
{

    BL_PROFILE("amr-wind::Sampling::sampling_post");

    for (const auto& obj : m_samplers) {
        obj->post_sample_actions();
    }

    m_output_buf.clear();
}

void Sampling::post_regrid_actions()
{
    BL_PROFILE("amr-wind::Sampling::post_regrid_actions");
    m_scontainer->Redistribute();
}

void Sampling::convert_velocity_lineofsight()
{
    BL_PROFILE("amr-wind::Sampling::convert_velocity_lineofsight");
    const long nvars = m_var_names.size();
    std::vector<int> vel_map;

    for (int iv = 0; iv < nvars; ++iv) {
        if (m_var_names[iv] == "velocityx") {
            vel_map.push_back(iv);
        }
        if (m_var_names[iv] == "velocityy") {
            vel_map.push_back(iv);
        }
        if (m_var_names[iv] == "velocityz") {
            vel_map.push_back(iv);
        }
    }

    AMREX_ALWAYS_ASSERT(static_cast<int>(vel_map.size()) == AMREX_SPACEDIM);

    int soffset = 0;
    for (const auto& obj : m_samplers) {
        long sample_size =
            obj->num_points(); // sample locs for individual sampler

        int scan_size =
            (obj->do_subsampling_interp()) ? sample_size / 2 : sample_size;

        std::vector<std::vector<double>> temp_vel(
            scan_size, std::vector<double>(AMREX_SPACEDIM));
        std::vector<std::vector<double>> temp_vel_next(
            scan_size, std::vector<double>(AMREX_SPACEDIM));

        if (obj->do_convert_velocity_los()) {
            for (int iv = 0; iv < AMREX_SPACEDIM; ++iv) {
                int vel_off = vel_map[iv];

                int offset =
                    vel_off * m_scontainer->num_sampling_particles() + soffset;
                for (int j = 0; j < scan_size; ++j) {
                    temp_vel[j][iv] = m_sample_buf[offset + j];
                    if (obj->do_subsampling_interp()) {
                        temp_vel_next[j][iv] =
                            m_sample_buf[scan_size + offset + j];
                    }
                }
            }
            obj->calc_lineofsight_velocity(temp_vel, 0);
            if (obj->do_subsampling_interp()) {
                obj->calc_lineofsight_velocity(temp_vel_next, 1);
            }
        }
        soffset += sample_size;
    }
}

void Sampling::create_output_buffer()
{
    BL_PROFILE("amr-wind::Sampling::create_output_buffer");
    const long nvars = m_var_names.size();
    for (int iv = 0; iv < nvars; ++iv) {
        int offset = iv * m_scontainer->num_sampling_particles();
        for (const auto& obj : m_samplers) {
            long sample_size = obj->num_points();
            if (obj->do_data_modification()) {
                // Run data through specific sampler's mod method
                const std::vector<double> temp_sb_mod(
                    &m_sample_buf[offset], &m_sample_buf[offset + sample_size]);
                std::vector<double> mod_result =
                    obj->modify_sample_data(temp_sb_mod, m_var_names[iv]);
                m_output_buf.insert(
                    m_output_buf.end(), mod_result.begin(), mod_result.end());
                offset += sample_size;
            } else {
                // Directly put m_sample_buf in m_output_buf
                std::vector<double> temp_sb(
                    &m_sample_buf[offset], &m_sample_buf[offset + sample_size]);
                m_output_buf.insert(
                    m_output_buf.end(), temp_sb.begin(), temp_sb.end());
                offset += sample_size;
            }
        }
    }

    m_output_particles = m_output_buf.size() / nvars;
}

void Sampling::fill_buffer()
{
    BL_PROFILE("amr-wind::Sampling::fill_buffer");
    m_scontainer->populate_buffer(m_sample_buf);
}

void Sampling::process_output()
{
    BL_PROFILE("amr-wind::Sampling::process_output");
    if (m_out_fmt == "native") {
        impl_write_native();
    } else if (m_out_fmt == "ascii") {
        write_ascii();
    } else if (m_out_fmt == "netcdf") {
        write_netcdf();
    } else {
        amrex::Abort("Sampling: Invalid output format encountered");
    }
}

void Sampling::impl_write_native()
{
    BL_PROFILE("amr-wind::Sampling::write_native");

    const std::string post_dir = "post_processing";
    const std::string name =
        amrex::Concatenate(m_label, m_sim.time().time_index());
    amrex::Vector<std::string> int_var_names{"uid", "set_id", "probe_id"};

    m_scontainer->WritePlotFile(
        post_dir, name, m_var_names, int_var_names,
        [=] AMREX_GPU_HOST_DEVICE(
            const SamplingContainer::SuperParticleType& p) {
            return p.id() > 0;
        });
}

void Sampling::write_ascii()
{
    BL_PROFILE("amr-wind::Sampling::write_ascii");
    amrex::Print() << "WARNING: Sampling: ASCII output will impact performance"
                   << std::endl;

    const std::string post_dir = "post_processing";
    const std::string sname =
        amrex::Concatenate(m_label, m_sim.time().time_index());

    if (!amrex::UtilCreateDirectory(post_dir, 0755)) {
        amrex::CreateDirectoryFailed(post_dir);
    }
    const std::string fname = post_dir + "/" + sname + ".txt";
    m_scontainer->WriteAsciiFile(fname);
}

void Sampling::prepare_netcdf_file()
{
#ifdef AMR_WIND_USE_NETCDF

    const std::string post_dir = "post_processing";
    const std::string sname =
        amrex::Concatenate(m_label, m_sim.time().time_index());
    if (!amrex::UtilCreateDirectory(post_dir, 0755)) {
        amrex::CreateDirectoryFailed(post_dir);
    }
    m_ncfile_name = post_dir + "/" + sname + ".nc";

    // Only I/O processor handles NetCDF generation
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    auto ncf = ncutils::NCFile::create(m_ncfile_name, NC_CLOBBER | NC_NETCDF4);
    const std::string nt_name = "num_time_steps";
    const std::string npart_name = "num_points";
    const std::vector<std::string> two_dim{nt_name, npart_name};
    ncf.enter_def_mode();
    ncf.put_attr("title", "AMR-Wind data sampling output");
    ncf.put_attr("version", ioutils::amr_wind_version());
    ncf.put_attr("created_on", ioutils::timestamp());
    ncf.def_dim(nt_name, NC_UNLIMITED);
    ncf.def_dim("ndim", AMREX_SPACEDIM);
    ncf.def_var("time", NC_DOUBLE, {nt_name});

    // Define groups for each sampler
    for (const auto& obj : m_samplers) {
        auto grp = ncf.def_group(obj->label());
        grp.def_dim(npart_name, obj->num_output_points());
        obj->define_netcdf_metadata(grp);
        grp.def_var("coordinates", NC_DOUBLE, {npart_name, "ndim"});

        // Create variables in each sampler
        // Removing velocity components when LOS velocity is output
        for (const std::string vname : m_var_names) {
            if (obj->do_convert_velocity_los() == false) {
                grp.def_var(vname, NC_DOUBLE, two_dim);
            } else {
                if (vname.find("velocity") == std::string::npos) {
                    grp.def_var(vname, NC_DOUBLE, two_dim);
                }
            }
        }

        if (obj->do_convert_velocity_los()) {
            grp.def_var("los_velocity", NC_DOUBLE, two_dim);
        }
    }
    ncf.exit_def_mode();

    {
        const std::vector<size_t> start{0, 0};
        std::vector<size_t> count{0, AMREX_SPACEDIM};
        SamplerBase::SampleLocType locs;
        for (const auto& obj : m_samplers) {
            auto grp = ncf.group(obj->label());
            obj->populate_netcdf_metadata(grp);
            obj->output_locations(locs);
            auto xyz = grp.var("coordinates");
            count[0] = obj->num_output_points();
            xyz.put(&locs[0][0], start, count);
        }
    }

#else
    amrex::Abort(
        "NetCDF support was not enabled during build time. Please recompile or "
        "use native format");
#endif
}

void Sampling::write_netcdf()
{
#ifdef AMR_WIND_USE_NETCDF
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    auto ncf = ncutils::NCFile::open(m_ncfile_name, NC_WRITE);
    const std::string nt_name = "num_time_steps";
    // Index of the next timestep
    const size_t nt = ncf.dim(nt_name).len();
    {
        auto time = m_sim.time().new_time();
        ncf.var("time").put(&time, {nt}, {1});
    }

    for (const auto& obj : m_samplers) {
        auto grp = ncf.group(obj->label());
        obj->output_netcdf_data(grp, nt);
    }

    std::vector<size_t> start{nt, 0};
    std::vector<size_t> count{1, 0};

    // Standard sampler output from input deck
    const int nvars = m_var_names.size();
    for (int iv = 0; iv < nvars; ++iv) {
        std::string vname = m_var_names[iv];
        start[1] = 0;
        count[1] = 0;
        int offset = iv * num_output_particles();
        for (const auto& obj : m_samplers) {
            auto grp = ncf.group(obj->label());
            count[1] = obj->num_output_points();

            if (obj->do_convert_velocity_los() == false) {
                auto var = grp.var(vname);
                var.put(&m_output_buf[offset], start, count);
            } else {
                if (vname.find("velocity") == std::string::npos) {
                    auto var = grp.var(vname);
                    var.put(&m_output_buf[offset], start, count);
                }
            }
            offset += count[1];
        }
    }

    // Custom sampler output from sampler function
    // Output of los_velocity goes here in addition to other custom output
    for (const auto& obj : m_samplers) {
        auto grp = ncf.group(obj->label());
        bool custom_output = obj->output_netcdf_field(m_output_buf, grp, nt);
    }

    ncf.close();
#endif
}

} // namespace amr_wind::sampling
