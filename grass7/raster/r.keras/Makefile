MODULE_TOPDIR = ../..

PGM = r.keras

# note: to deactivate a module, just place a file "DEPRECATED" into the subdir
SUBDIRS = r.keras.class

include $(MODULE_TOPDIR)/include/Make/Dir.make

default: parsubdirs htmldir

install: installsubdirs
	$(INSTALL_DATA) $(PGM).html $(INST_DIR)/docs/html/
