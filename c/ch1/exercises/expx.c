#include <petsc.h>

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i;
  PetscReal      localval, globalsum, x;

  ierr = PetscInitialize(&argc,&argv,NULL,
      "Compute exp in parallel with PETSc.\n\n"); if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for expx", "");
  PetscOptionsReal("-x", "input to exp(x) function", "expx.c", x, &x, PETSC_NULL);
  PetscOptionsEnd();

  // compute  exp(x)  where n = (rank of process) + 1
  localval = 1.0;
  for (i = 2; i < rank+1; i++)
      localval /= i;
  localval *= PetscPowReal(x, rank);

  // sum the contributions over all processes
  ierr = MPI_Allreduce(&localval,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD); CHKERRQ(ierr);

  // output estimate of e and report on work from each process
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "exp(%f) is about %17.15f\n", x, globalsum); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,
      "rank %d did %d flops\n", rank, (rank > 0) ? rank-1 : 0);
      CHKERRQ(ierr);
  return PetscFinalize();
}
