#include <iostream>
#include <algorithm>

#include <ocl_wrapper.h>
#include <utl_utils.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

typedef float Type;
typedef utl::Matrix <Type,utl::column_major_tag> Matrix;
typedef utl::Ones <Type,utl::column_major_tag> Ones;
typedef utl::Zeros <Type,utl::column_major_tag> Zeros;
typedef utl::Rand <Type,utl::column_major_tag, utl::uniform_dist_tag> Rand;

int main(int argc, char* argv[])
{

		// provides a convient access to the command line arguments
		utl::Args args(argc, argv);
	/**	
    const size_t elements_in = args.size() > 1 ? args.toSizet(1) : 1 << 22;
    const size_t size_bytes_in  = elements_in * sizeof(Type);
    const size_t local_size = 256;
    const size_t elements_out = std::max(elements_in/local_size, 1ul);
		const size_t size_bytes_out = elements_out * sizeof(Type);
        */
    const size_t execute = 100;
			
		

    ocl::Platform platform(ocl::device_type::GPU);
    ocl::Device device = platform.device(ocl::device_type::GPU);
    // creates a context for a decice or platform
    ocl::Context context(device);
    // inserts contexts into the platform
    platform.insert(context);
    // creates command queue.
    ocl::Queue queue(context, device);
    // create program on a context
    // as the kernel is templated, creates kernel for single and integer types
    ocl::Program program(context, utl::type::Single | utl::type::Int);       	
    // inserts kernels into the program.
    std::ifstream file("11.multi/multi.cl");
    program << file;
    // kernels are created and program is built for the context.
    program.build();

    ocl::Kernel &kernel = program.kernel("multi", utl::type::Single);

    // set the dimensions
    size_t rows = 1<<11, cols = 1<<12, common = 1<<12;

    size_t elements_A = rows*common;
    size_t elements_B = common*cols;
    size_t elements_C = rows * cols;
    size_t size_bytes_A = elements_A * sizeof(Type);
    size_t size_bytes_B = elements_B * sizeof(Type);
    size_t size_bytes_C = elements_C * sizeof(Type);

    kernel.setWorkSize(16,16, rows, cols);

    //create host matrices
    auto h_matrix_A  = Rand(rows,common);
    auto h_matrix_B  = Rand(common, cols);
    auto h_matrix_C = Zeros(rows,cols);

    //std::cout << "Matrix(A) before computation: " << std::endl << h_matrix_A << std::endl;
    //std::cout << "Matrix(B) before computation: " << std::endl << h_matrix_B << std::endl;
    //std::cout << "Matrix(C) before computation: " << std::endl << h_matrix_C << std::endl;

    //create Buffers on device
    ocl::Buffer d_matrix_A (context, size_bytes_A);
    ocl::Buffer d_matrix_B (context,size_bytes_B);
    ocl::Buffer d_matrix_C (context, size_bytes_C);

    // copy data from host mem to buffers
    d_matrix_A.write(queue, 0, h_matrix_A.data(), size_bytes_A);
    d_matrix_B.write(queue, 0, h_matrix_B.data(), size_bytes_B);
    d_matrix_C.write(queue, 0, h_matrix_C.data(), size_bytes_C);

    //timer start
    utl::Timer::tic();

    //call kernel
    kernel(queue, int(rows), int(cols), int(common), d_matrix_A.id(), d_matrix_B.id(), d_matrix_C.id());
    queue.finish();

    //timer end
    utl::Timer::toc();

    //write result from device buffer to host mem
    d_matrix_C.read(queue, h_matrix_C.data(), size_bytes_C);

    std::cout << "Time elapsed: " << utl::Seconds(utl::Timer::elapsed(execute)) << std::endl;
    //std::cout << "Matrix(C) after computation : " << std::endl << "A = " << h_matrix_C << std::endl;


	return 0;
}
