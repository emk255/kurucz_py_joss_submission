FC ?= ifort
ifeq ($(findstring gfortran,$(FC)),gfortran)
# GNU Fortran: avoid ifort-only flags, keep fixed-form and static locals
FFLAGS = -c -fno-automatic -O0 -g -ffixed-line-length-none -fbacktrace
LDFLAGS =
else
# Intel Fortran: original flags retained
FFLAGS = -c -save -1 -w -V -WB -double-size 64 -fp-model strict
LDFLAGS = -zero
endif

all: atlas12 install

atlas12:
	$(FC) $(FFLAGS)  -o atlas7lib.o  atlas7v.for
	@echo "Compiled atlas7v.for -> atlas7lib.o"

	$(FC) $(FFLAGS) -o at12tosyn.o at12tosyn.f90
	$(FC) $(LDFLAGS) -o at12tosyn.exe at12tosyn.o

	$(FC) $(FFLAGS) -o rgfall.o rgfall.for
	$(FC) $(LDFLAGS)  -o rgfall.exe rgfall.o

	$(FC) $(FFLAGS) -o spectrv.o spectrv.for
	$(FC) $(LDFLAGS) -o spectrv.exe spectrv.o atlas7lib.o

	$(FC) $(FFLAGS)  -o synbeg.o synbeg.for
	$(FC) $(LDFLAGS)  -o synbeg.exe synbeg.o

	$(FC) $(FFLAGS)  -o synthe.o synthe.for
	$(FC) $(LDFLAGS) -o synthe.exe synthe.o

	$(FC) $(FFLAGS) -o syntoascanga.o syntoascanga.for
	$(FC) $(LDFLAGS)  -o syntoascanga.exe syntoascanga.o

	$(FC) $(FFLAGS) -o xnfpelsyn.o xnfpelsyn.for
	$(FC) $(LDFLAGS) -o xnfpelsyn.exe xnfpelsyn.o atlas7lib.o

install:	
	/bin/mv *.exe ../bin
	/bin/rm -f *.o

