#ifndef CHANNELSTATSBASE_H
#define CHANNELSTATSBASE_H

#include "amr-wind/core/Factory.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/FieldPlaneAveraging.H"

namespace amr_wind {

class ChannelStatsBase
    : public Factory<ChannelStatsBase, CFDSim&, int>
{
public:
    static std::string base_identifier() { return "ChannelStatsBase"; }

    ChannelStatsBase() = default;

    virtual ~ChannelStatsBase() = default; 

    //! Interpolating object for vertical velocity profile
    virtual const VelPlaneAveraging& vel_profile() const = 0;

    //! Perform initialization actions after the mesh has been created
    virtual void post_init_actions() = 0;

    //! Perform actions at the beginning of a timestep
    virtual void pre_advance_work() = 0;

    //! Perform actions at the end of a timestep
    virtual void post_advance_work() = 0;
};

} // namespace amr_wind

#endif /* CHANNELSTATSBASE_H */
