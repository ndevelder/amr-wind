#include <AMReX_REAL.H>
#include <AMReX_BLFort.H>
#include <AMReX_SPACE.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#endif

#include <AMReX_MultiFabUtil.H>
#include <AMReX_MacProjector.H>
#include <incflo_proj_F.H>

#include <incflo.H>

using namespace amrex;

//
// Computes the following decomposition:
// 
//    u + c*grad(phi)/ro = u*  with  div(ep*u) = 0
//
// Inputs:
// 
//   u_mac,v_mac,w_mac = the MAC velocity field to be projected
//   density           = the cell-centered density
//
// Outputs:
//
//  u_mac,v_mac,w_mac = the PROJECTED MAC velocity field 
//
// Notes:
//
//  phi, the projection auxiliary function, is computed by solving
//
//       div(ep*grad(phi)/rho) = div(ep * u*)
// 
void 
incflo::apply_MAC_projection (Vector<MultiFab>& u_mac,
                              Vector<MultiFab>& v_mac,
                              Vector<MultiFab>& w_mac,
                              Vector<MultiFab const*> const& density,
                              Real time)
{
    BL_PROFILE("incflo::apply_MAC_projection()");

    if (incflo_verbose > 0) amrex::Print() << "MAC Projection:\n";

    // This will hold (1/rho) on faces
    Vector<Array<MultiFab ,AMREX_SPACEDIM> > rho_face(finest_level+1);
    Vector<Array<MultiFab*,AMREX_SPACEDIM> > mac_vec(finest_level+1);
    for (int lev=0; lev <= finest_level; ++lev)
    {
        rho_face[lev][0].define(u_mac[lev].boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));
        rho_face[lev][1].define(v_mac[lev].boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));
        rho_face[lev][2].define(w_mac[lev].boxArray(),dmap[lev],1,0,MFInfo(),Factory(lev));

        amrex::average_cellcenter_to_face(GetArrOfPtrs(rho_face[lev]), *density[lev], geom[lev]);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            rho_face[lev][idim].invert(1.0, 0);
        }

        mac_vec[lev][0] = &u_mac[lev];
        mac_vec[lev][1] = &v_mac[lev];
        mac_vec[lev][2] = &w_mac[lev];
    }

    //
    // If we want to set max_coarsening_level we have to send it in to the constructor
    //
    LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(mac_mg_max_coarsening_level);

    //
    // Perform MAC projection
    //
    MacProjector macproj(mac_vec, GetVecOfArrOfConstPtrs(rho_face), Geom(0,finest_level), lp_info);

    macproj.setDomainBC(get_projection_bc(Orientation::low), get_projection_bc(Orientation::high));

    macproj.project(mac_mg_rtol,mac_mg_atol,MLMG::Location::FaceCentroid);
}

void
incflo::apply_MAC_projection (Vector< std::unique_ptr<MultiFab> >& u_mac, 
                              Vector< std::unique_ptr<MultiFab> >& v_mac,
                              Vector< std::unique_ptr<MultiFab> >& w_mac,
                              Vector< std::unique_ptr<MultiFab> >& density_in,
                              Real time)
{
}
