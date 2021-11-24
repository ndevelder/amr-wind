#ifndef ChannelFlowLES_H
#define ChannelFlowLES_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/Field.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/physics/ChannelStatsBase.H"

namespace amr_wind {
namespace channel_flow_les {

/** Channel Flow physics
 *  \ingroup physics
 */
class ChannelFlowLES : public Physics::Register<ChannelFlowLES>
{
public:
    static const std::string identifier() { return "ChannelFlowLES"; }

    explicit ChannelFlowLES(CFDSim& sim);

    virtual ~ChannelFlowLES() = default;

    void initialize_fields(int level, const amrex::Geometry& geom) override;

    //! Use Reichardt function to initialize velocity profile
    template <typename IndexSelector>
    void initialize_fields(
        int level,
        const amrex::Geometry& geom,
        const IndexSelector& idxOp,
        const int n_idx);

    void post_init_actions() override;

    void post_regrid_actions() override {}

    void pre_advance_work() override;

    void post_advance_work() override;

    //! Return the statistics calculator
    const ChannelStatsBase& channel_statistics() const { return *m_stats; }

private:
    const amr_wind::SimTime& m_time;
    const FieldRepo& m_repo;
    const amrex::AmrCore& m_mesh;

    //! Wall normal directon - Default y direction
    int m_norm_dir{1};

    //! initial density value
    amrex::Real m_rho{1.0};

    //! perturb scale 
    amrex::Real m_perturb_scale{1.0};

    //! perturb scale y
    amrex::Real m_perturb_scale_y{0.0};

    //! Re_tau
    amrex::Real m_re_tau{392.72};

    //! u_tau
    amrex::Real m_utau{1.0};

    //! y_tau
    amrex::Real m_ytau{m_re_tau};

    //! Von-Karman constant
    amrex::Real m_kappa{0.41};

    //! Number of sinusoidal waves in x-direction
    amrex::Real m_Uperiods{8.0};

    //! Number of sinusoidal waves in y-direction
    amrex::Real m_Vperiods{8.0};

    //! Number of sinusoidal waves in w-direction
    amrex::Real m_Wperiods{8.0};

    //! ABL integrated statistics object
    std::unique_ptr<ChannelStatsBase> m_stats;

    //! Perturb initial velocity field with sinusoidal fluctuations
    bool m_perturb_vel{true};
};
} // namespace channel_flow
} // namespace amr_wind

#endif /* ChannelFlowLES_H */
