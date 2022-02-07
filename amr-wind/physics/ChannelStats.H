#ifndef CHANNELISTATS_H
#define CHANNELISTATS_H

#include "amr-wind/physics/ChannelStatsBase.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/FieldPlaneAveraging.H"
#include "amr-wind/utilities/SecondMomentAveraging.H"
#include "amr-wind/utilities/ThirdMomentAveraging.H"
#include "amr-wind/utilities/PostProcessing.H"
#include "amr-wind/utilities/sampling/SamplerBase.H"
#include "amr-wind/utilities/sampling/SamplingContainer.H"

namespace amr_wind {

/**
 *  \defgroup channel_istats Channel Statistics
 *
 *  ChannelStats contains functions to compute and statistics for Channel Flow
 *  simulations. It supports output in ascii format as well as NetCDF format.
 *
 * \ingroup we_channel
 */
class ChannelStats : public ChannelStatsBase::Register<ChannelStats>
{
public:
    static const std::string identifier() { return "precursor"; }

    ChannelStats(CFDSim&, const int dir);

    virtual ~ChannelStats();

    // Perform initialization actions after the mesh has been created
    void post_init_actions() override;

    //! Perform actions before a new timestep
    void pre_advance_work() override {}

    //! Process fields given timestep and output to disk
    void post_advance_work() override;

    //! Return vel plane averaging instance
    const VelPlaneAveraging& vel_profile() const override { return m_pa_vel; };

    //! Calculate sfs stress averages
    void
    calc_sfs_stress_avgs(ScratchField& sfs_stress);

protected:
    //! Read user inputs and create the necessary files
    void initialize();

    //! Calculate plane average profiles
    void calc_averages();

    //! Output data based on user-defined format
    virtual void process_output();

    //! Prepare ASCII file
    virtual void prepare_ascii_file();
    //! Prepare NetCDF metadata
    virtual void prepare_netcdf_file();

    //! Write sampled data into a NetCDF file
    void write_netcdf();

    /** Output sampled data in ASCII format
     *
     *  Note that this should be used for debugging only and not in production
     *  runs as it can have significant impacts on code performance.
     */
    virtual void write_ascii();

private:
    CFDSim& m_sim;
    Field& m_mueff;

    VelPlaneAveraging m_pa_vel;
    FieldPlaneAveraging m_pa_mueff;
    SecondMomentAveraging m_pa_uu;
    ThirdMomentAveraging m_pa_uuu;

    //! Format of the data output (ascii, netcdf, etc.)
#ifdef AMR_WIND_USE_NETCDF
    std::string m_out_fmt{"netcdf"};
    std::string m_ncfile_name;
#else
    std::string m_out_fmt{"ascii"};
#endif
    std::string m_ascii_file_name;

    //! Frequency of data sampling and output
    int m_out_freq{100};

    //! Acceleration due to gravity magnitude
    amrex::Real m_gravity{9.81};

    //! Von-Karman constant
    amrex::Real m_kappa{0.41};

    //! Variable to store capping inversion height
    double m_zi{0.0};

    //! Wall-normal direction axis
    int m_normal_dir{2};

    //! Cell spacing at the coarsest level
    double m_dn{0.0};

    //! Number of cells in the horizontal direction
    size_t m_ncells_h1{0};
    size_t m_ncells_h2{0};
};

} // namespace amr_wind

#endif /* CHANNELISTATS_H */
