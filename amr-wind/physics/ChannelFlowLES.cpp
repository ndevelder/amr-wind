#include "amr-wind/physics/ChannelFlowLES.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_iMultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "amr-wind/utilities/trig_ops.H"
#include "amr-wind/utilities/DirectionSelector.H"

namespace amr_wind {
namespace channel_flow_les {

ChannelFlowLES::ChannelFlowLES(CFDSim& sim)
    : m_time(sim.time()), m_repo(sim.repo()), m_mesh(sim.mesh())
{

    {
        amrex::ParmParse pp("ChannelFlowLES");
        pp.query("normal_direction", m_norm_dir);

        pp.query("density", m_rho);
        pp.query("re_tau", m_re_tau);
        pp.query("perturb_velocity", m_perturb_vel);
        pp.query("Uperiods", m_Uperiods);
        pp.query("Vperiods", m_Vperiods);
        pp.query("Wperiods", m_Wperiods);
        pp.query("perturb_scale", m_perturb_scale);
        pp.query("perturb_scale_y", m_perturb_scale_y);
    }

    {
        amrex::Real mu;
        amrex::ParmParse pp("transport");
        pp.query("viscosity", mu);
        // Assumes a boundary layer height of 1.0
        m_utau = mu * m_re_tau / (m_rho * 1.0);
        m_ytau = mu / (m_utau * m_rho);
    }

    {
        std::string statistics_mode = "precursor";
        int dir = 2;
        amrex::ParmParse pp("ChannelFlowLES");
        pp.query("normal_direction", dir);
        pp.query("statistics_mode", statistics_mode);
        m_stats =
            ChannelStatsBase::create(statistics_mode, sim, dir);
    }
}

/** Initialize the velocity, density fields at the beginning of the
 *  simulation.
 */
void ChannelFlowLES::initialize_fields(int level, const amrex::Geometry& geom)
{

    switch (m_norm_dir) {
    case 1:
        initialize_fields(level, geom, YDir(), 1);
        break;
    case 2:
        initialize_fields(level, geom, ZDir(), 2);
        break;
    default:
        amrex::Abort("axis must be equal to 1 or 2");
        break;
    }
}

template <typename IndexSelector>
void ChannelFlowLES::initialize_fields(
    int level,
    const amrex::Geometry& geom,
    const IndexSelector& idxOp,
    const int n_idx)
{  

    const amrex::Real kappa = m_kappa;
    const amrex::Real y_tau = m_ytau;
    const amrex::Real utau = m_utau;
    auto& velocity = m_repo.get_field("velocity")(level);
    auto& density = m_repo.get_field("density")(level);

    density.setVal(m_rho);

    for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        const auto& probhi = geom.ProbHiArray();
        auto vel = velocity.array(mfi);
        const bool perturb_vel = m_perturb_vel;

        const amrex::Real pi = M_PI;
        const amrex::Real per_scale = m_perturb_scale;
        const amrex::Real per_scale_y = m_perturb_scale_y;
        const amrex::Real aval = m_Uperiods * 2.0 * pi / (probhi[0] - problo[0]);
        const amrex::Real bval = m_Vperiods * 2.0 * pi / (probhi[1] - problo[1]);
        const amrex::Real cval = m_Wperiods * 2.0 * pi / (probhi[2] - problo[2]);


        // Currently assumes a channel half height of 1.0m
        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
                
                const int n_ind = idxOp(i, j, k);
                amrex::Real h = problo[n_idx] + (n_ind + 0.5) * dx[n_idx];
                if (h > 1.0) h = 2.0 - h;
                const amrex::Real hp = h / y_tau;
                vel(i, j, k, 0) =
                    utau * (1. / kappa * std::log1p(kappa * hp) +
                            7.8 * (1.0 - std::exp(-hp / 11.0) -
                                   (hp / 11.0) * std::exp(-hp / 3.0)));
                
                vel(i, j, k, 1) = 0.0;
                vel(i, j, k, 2) = 0.0;

                
                
                if (perturb_vel) {
                    const amrex::Real xl = x - problo[0];
                    const amrex::Real yl = y - problo[1];
                    const amrex::Real zl = z - problo[2];

                    amrex::Real yw = yl;
                    if (yw > ((probhi[1] - problo[1])/2.0)) yw = (probhi[1] - problo[1]) - yw;

                    amrex::Real zw = zl;
                    if (zw > ((probhi[2] - problo[2])/2.0)) zw = (probhi[2] - problo[2]) - zw;

                    const amrex::Real damp = 1.0 - std::exp(-6.0 * zw * zw);
                    //const amrex::Real dampy = 1.0 - std::exp(-3.0 * yw * yw * yw * yw);
                    vel(i, j, k, 0) += per_scale * damp * std::cos(aval * yl);
                    vel(i, j, k, 1) += per_scale_y * damp * std::cos(bval * xl);
                    // vel(i, j, k, 2) += per_scale * damp * std::cos(cval * zl);
                }
            });
    }
}

void ChannelFlowLES::post_init_actions() 
{
    m_stats->post_init_actions();
}

void ChannelFlowLES::pre_advance_work()
{
    const auto& vel_pa = m_stats->vel_profile();
}

void ChannelFlowLES::post_advance_work() 
{
    m_stats->post_advance_work();
}

} // namespace channel_flow
} // namespace amr_wind
