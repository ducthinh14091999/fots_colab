CXXFLAGS = -I include -std=c++11 -O3 $(shell python3-config --cflags)
CXXFLAGS := $(filter-out -fno-plt -mtune=haswell, $(CXXFLAGS))
# LDFLAGS = $(shell python37-config --ldflags)
include_path_link= -I C:\Users\thinh\anaconda3\include
DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.pyd

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS)  $(CXX_SOURCES) --shared -fPIC -Wno-deprecated $(include_path_link) -Wclass-memaccess -Wunused-variable

# cl adaptor.cpp .\include\clipper\clipper.cpp /I /include /I "C:/Users/thinh/anaconda3/include" /LD /Fe:adapter.pyd /link/LIBPATH:  "C:/Users/thinh/anaconda3/libs"
# $(LDFLAGS)
# clean:
# 	rm -rf $(LIB_SO)
