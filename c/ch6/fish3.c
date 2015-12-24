static char help[] = "Solves a 3D structured-grid Poisson problem with DMDA\n"
"and SNES.\n\n";

/* in this version, these work:
  ./fish3 -snes_fd
*/

#include <petsc.h>

typedef struct {
  DM        da;
  PetscReal hx, hy, hz;
  Vec       f;
} Ctx;


PetscErrorCode formRHS(DMDALocalInfo *info, Ctx *usr) {
    PetscErrorCode ierr;
    PetscInt       i, j, k;
    //PetscReal      x, y, z;
    PetscReal      ***af;

    ierr = DMDAVecGetArray(usr->da, usr->f, &af);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        //z = k * usr->hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            //y = j * usr->hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                //x = i * usr->hx;
                af[k][j][i] = 1.0;
            }
        }
    }
    ierr = DMDAVecRestoreArray(usr->da, usr->f, &af);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(usr->f); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(usr->f); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal ***u,
                                 PetscReal ***F, Ctx *usr) {
    PetscErrorCode  ierr;
    PetscInt        i, j, k;
    PetscReal       uxx, uyy, uzz, ***af;
    const PetscReal hx2 = usr->hx*usr->hx,
                    hy2 = usr->hy*usr->hy,
                    hz2 = usr->hz*usr->hz;

    ierr = DMDAVecGetArray(usr->da, usr->f, &af);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        for (j=info->ys; j<info->ys+info->ym; j++) {
            for (i=info->xs; i<info->xs+info->xm; i++) {
                if (i == 0 || j == 0 || k == 0
                    || i == info->mx-1 || j == info->my-1 || k == info->mz-1) {
                    F[k][j][i] = u[k][j][i];
                } else {
                    uxx = (u[k][j][i-1] - 2.0 * u[k][j][i] + u[k][j][i+1]) / hx2;
                    uyy = (u[k][j-1][i] - 2.0 * u[k][j][i] + u[k][j+1][i]) / hy2;
                    uzz = (u[k-1][j][i] - 2.0 * u[k][j][i] + u[k+1][j][i]) / hz2;
                    F[k][j][i] = - uxx - uyy - uzz - af[k][j][i];
                }
            }
        }
    }
    ierr = DMDAVecRestoreArray(usr->da, usr->f, &af);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar ***u,
                                 Mat J, Mat Jpre, Ctx *usr) {

    PetscErrorCode  ierr;
    PetscInt        i,j,k,q;
    PetscReal       v[7];
    MatStencil      col[7],row;
    const PetscReal hx2 = usr->hx*usr->hx,
                    hy2 = usr->hy*usr->hy,
                    hz2 = usr->hz*usr->hz,
                    diag = 2.0*(1.0/hx2 + 1.0/hy2 + 1.0/hz2);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        row.k = k;
        col[0].k = k;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            row.j = j;
            col[0].j = j;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                row.i = i;
                col[0].i = i;
                q = 1;   // will insert at least one element
                if (i == 0 || j == 0 || k == 0
                        || i == info->mx-1 || j == info->my-1 || k == info->mz-1) {
                    v[0] = 1.0;
                } else {
                    v[0] = diag;
                    if (i-1 != 0) {
                        v[q] = - 1.0/hx2;
                        col[q].k = k;  col[q].j = j;  col[q].i = i-1;
                        q++;
                    }
                    if (i+1 != info->mx-1) {
                        v[q] = - 1.0/hx2;
                        col[q].k = k;  col[q].j = j;  col[q].i = i+1;
                        q++;
                    }
                    if (j-1 != 0) {
                        v[q] = - 1.0/hy2;
                        col[q].k = k;  col[q].j = j-1;  col[q].i = i;
                        q++;
                    }
                    if (j+1 != info->my-1) {
                        v[q] = - 1.0/hy2;
                        col[q].k = k;  col[q].j = j+1;  col[q].i = i;
                        q++;
                    }
                    if (k-1 != 0) {
                        v[q] = - 1.0/hz2;
                        col[q].k = k-1;  col[q].j = j;  col[q].i = i;
                        q++;
                    }
                    if (k+1 != info->mz-1) {
                        v[q] = - 1.0/hz2;
                        col[q].k = k+1;  col[q].j = j;  col[q].i = i;
                        q++;
                    }
                }
                ierr = MatSetValuesStencil(Jpre,1,&row,q,col,v,INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    return 0;
}

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    SNES           snes;
    Vec            u;
    DMDALocalInfo  info;
    Ctx            user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,
                -5,-5,-5,
                PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                1,1,
                NULL,NULL,NULL,
                &user.da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(user.da,0.0,1.0,0.0,1.0,0.0,1.0); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
    user.hx = 1.0/(info.mx-1);
    user.hy = 1.0/(info.my-1);
    user.hz = 1.0/(info.mz-1);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(user.da,
            (DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(user.da,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&user.f); CHKERRQ(ierr);
    ierr = formRHS(&info,&user); CHKERRQ(ierr);

    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    //ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
    //ierr = formExact(user.da,uexact); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d x %d grid ...\n",
             info.mx,info.my,info.mz); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&user.f);
    SNESDestroy(&snes);  DMDestroy(&user.da);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}

