#include <AMReX_ParmParse.H>

#include <algorithm>
#include <incflo.H>

using namespace amrex;

void incflo::MakeEBGeometry()
{
   /******************************************************************************
   * incflo.geometry=<string> specifies the EB geometry. <string> can be one of    *
   * box, cylinder, annulus, sphere, spherecube, twocylinders
   ******************************************************************************/

    ParmParse pp("incflo");

    std::string geom_type;
    pp.query("geometry", geom_type);

   /******************************************************************************
   *                                                                            *
   *  CONSTRUCT EB                                                              *
   *                                                                            *
   ******************************************************************************/

    if(geom_type == "box")
    {
	amrex::Print() << "\n Building box geometry." << std::endl;
        make_eb_box();
    }
    else if(geom_type == "cylinder")
    {
	amrex::Print() << "\n Building cylinder geometry." << std::endl;
        make_eb_cylinder();
    }
    else if(geom_type == "twocylinders")
    {
	amrex::Print() << "\n Building twocylinders geometry." << std::endl;
        make_eb_twocylinders();
    }
    else if(geom_type == "annulus")
    {
	amrex::Print() << "\n Building annulus geometry." << std::endl;
        make_eb_annulus();
    }
    else if(geom_type == "sphere")
    {
	amrex::Print() << "\n Building sphere geometry." << std::endl;
        make_eb_sphere();
    }
    else if(geom_type == "spherecube")
    {
	amrex::Print() << "\n Building spherecube geometry." << std::endl;
        make_eb_spherecube();
    }
    else if(geom_type == "tuscan")
    {
	amrex::Print() << "\n Building tuscan geometry." << std::endl;
        make_eb_tuscan();
    }
    else
    {
	amrex::Print() << "\n No EB geometry declared in inputs => "
	               << " Will read walls from incflo.dat only." << std::endl;
        make_eb_regular();
    }
    amrex::Print() << "Done making the geometry ebfactory.\n" << std::endl;
}

// This function checks if ebfactory is allocated with
// the proper dm and ba
bool incflo::UpdateEBFactory(int a_lev)
{
#if 0
    // This assert is to verify that some kind of EB geometry
    // has already been defined
    AMREX_ASSERT(not EB2::IndexSpace::empty());

    const DistributionMapping& dm = DistributionMap(a_lev);
    const BoxArray&            ba = boxArray(a_lev);
    const EB2::IndexSpace&   ebis = EB2::IndexSpace::top();
    const EB2::Level&  ebis_level = ebis.getLevel(geom[a_lev]);

    bool is_updated = false;

    EBSupport eb_support_level = EBSupport::full;
    if ( ebfactory[a_lev] == nullptr
         or ebfactory[a_lev]->DistributionMap() != dm
         or ebfactory[a_lev]->boxArray() != ba )
    {
        ebfactory[a_lev].reset(new EBFArrayBoxFactory(ebis_level, geom[a_lev], ba, dm,
                                                      {m_eb_basic_grow_cells,
                                                       m_eb_volume_grow_cells,
                                                       m_eb_full_grow_cells},
                                                       eb_support_level));
        is_updated = true;
    }

    return is_updated;
#endif
    return true;
}
