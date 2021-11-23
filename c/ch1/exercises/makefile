include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

expx: expx.o
	-${CLINKER} -o expx expx.o  ${PETSC_LIB}
	${RM} expx.o

# testing
rune_1:
	-@../testit.sh expx "" 1 1

test_expx: rune_1

test: test_expx

.PHONY: distclean rune_1 test_expx test

distclean:
	@rm -f *~ expx *tmp

