
CFILES  = $(wildcard *.cpp)
OBJS1   = $(notdir $(CFILES))
OBJS2   = $(patsubst %.cpp,%.o, $(OBJS1))
OBJS    = $(addprefix build/,$(OBJS2))	


TARGET := ../multi

$(TARGET): $(OBJS)
		g++ $(GCC_FLAGS) $(OBJS) $(LIBS) -o $(TARGET)

build/%.o : %.cpp create_build_dir
	$(CC) -c $(INCS) $(GCC_FLAGS) $< -o $@

create_build_dir:
	mkdir -p ./build

.PHONY : clean

clean:
	rm -f build/*  $(TARGET)

