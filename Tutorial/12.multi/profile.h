#ifndef STUD4PASS_H
#define STUD4PASS_H

#include <iostream>
#include <stdexcept>
#include <memory>
#include <istream>

#include <ocl_wrapper.h>
#include <utl_utils.h>


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

/*! Stud4Pass is a Pass and managed by the PassManager.
 *
 *
 * \param Type_ is the value type of the matrices. Here we use float,double or int. other types are also possible
 * \param Format_ is the storage type of the matrices. Here we use row or column-major format.
*/
template <class Type_,class Format_ , size_t W1, size_t W2>
class Stud4Pass1 : public utl::ProfilePass
{
	using Base   = utl::ProfilePass;
	using Type   = Type_;
	using Format = Format_;
	using Rand   = utl::Rand  < Type, Format, utl::uniform_dist_tag >;
	using Zeros  = utl::Zeros < Type, Format >;
	using Ones   = utl::Ones  < Type, Format >;
	using Matrix = utl::Matrix< Type, Format >;
	using Dim    = utl::Dim;
public :

	Stud4Pass1() = delete;
	Stud4Pass1(const Stud4Pass1&) = default;
	Stud4Pass1(Stud4Pass1&&) = default;
	~Stud4Pass1() = default;


	/*! This is the constructor one should use to initialize the platform. */
	Stud4Pass1(const std::string& filename,   /*! Name of the *.cl file */
			   const std::string& kernelname, /*! Kernel name within the *.cl file */
			   const Dim& start,              /*! First dimension e.g. Dim(128,128,128) with Dim[0]=M, Dim[1]=N, Dim[2]=K, */
			   const Dim& step,               /*! Step dimension e.g. Dim(32,32,32) such that this pass iterates from first to last dimension */
			   const Dim& end,                /*! Last dimension e.g. Dim(256,256,256) with Dim[0]=M, Dim[1]=N, Dim[2]=K, */
			   bool testing = false,          /*! If true, compares the cpu reference result to the gpu result */
			   size_t iter = 10);             /*! Number of kernel iterations */

	/*! This function needs to be defined so that it can be called from the pass manager. */
	utl::Seconds prof( Dim const& ) override;

	/*! This function needs to be defined so that it can be called from the pass manager. */
	double ops( Dim const& ) override;

private :

	std::string name(const std::string& kernel) const
	{
		std::ostringstream oss;
		oss << "stud4_" << kernel << "_" << utl::Type::type<Type>().name() <<  "_B" << W1 << "x" << W2;
		return oss.str();
	}


	bool testing_;
	ocl::Platform platform_; /*! Platform is selected here as GPU. Initialized in the constructor */
	ocl::Device   device_;   /*! The first Device is chosen. Initialized in the constructor */
	ocl::Context  context_;  /*! Only one Context is created. Initialized in the constructor */
	ocl::Queue    queue_;    /*! Only one Queue is created with the above Context and Device. Initialized in the constructor */
	ocl::Program  program_;  /*! Program is created in the constructor but built in the prof() function with dimension parameters.*/
	ocl::Kernel*  kernel_;   /*! Kernel is created in the constructor but built in the prof() function. */
};


/*! Constructor which is invoked in the main file and loaded into the PassManager.
 *
 * You do not have to change the constructor definition.
*/
template <class Type_,class Format_, size_t W1, size_t W2>
Stud4Pass1<Type_,Format_,W1,W2>::Stud4Pass1(
		const std::string& file,
		const std::string& kernel,
		const utl::Dim& start,
		const utl::Dim& step,
		const utl::Dim& end,
		bool testing,
		size_t iter) :
	  Base(this->name(kernel), start, step, end, testing ? 1 : iter),
	  testing_(testing),
	  platform_( ocl::device_type::GPU ),
	  device_( platform_.device( ocl::device_type::GPU ) ),
	  context_( device_ ),
	  queue_( context_, device_, CL_QUEUE_PROFILING_ENABLE ),
	  program_( context_, utl::type::Single | utl::type::Double ),
	  kernel_(nullptr)
{
	std::ifstream stream( file );
	if ( !stream.is_open() ) { throw std::runtime_error("Failed opening file " + file);}
	program_ << stream;

	kernel_ = &program_.kernel(kernel, utl::Type::type<Type_>());
	if ( kernel_ == nullptr ) { throw std::runtime_error( "kernel not valid" ); }
}


/*! Profile function of the Pass. This file must be changed according to your needs.
 *
 * \note It shall alse provide a testing possibility.
 * \note You do not have to invoke this function. The PassManager does everything.
 *       Just be sure to call the this->call(f) function. See below.
 *
 * \param dim Dimension which is between the first and the last.
*/
template <class Type_,class Format_ , size_t W1, size_t W2>
utl::Seconds Stud4Pass1<Type_,Format_, W1,W2>::prof( utl::Dim const& dim )
{

	  const size_t M = dim[0];
	  const size_t N = dim[1];
	  const size_t K = dim[2];

	  static_assert(W1 >= 8, "W1 < 8");

	  if( N <= 0 ) throw std::runtime_error( "N should be greater 0." );
	  if( M <= 0 ) throw std::runtime_error( "M should be greater 0." );
	  if( K <= 0 ) throw std::runtime_error( "K should be greater 0." );

	  std::ostringstream oss;
	  oss << "-w -Werror" << " -D M=" << M << "u -D N=" << N << "u -D K=" << K << "u -D W1=" << W1 << "u -D W2=" << W2 <<'u';

	  program_.setCompileOption( ocl::compile_option::FAST_MATH | ocl::compile_option::NO_SIGNED_ZERO | ocl::CompileOption( oss.str() ) );
	  program_.build();
	  if ( ! program_.isBuilt() ) { throw std::runtime_error( "program not built" ); }
	  if ( ! kernel_->created() ) { throw std::runtime_error( "kernel not created" ); }



	  kernel_->setWorkSize( W1, W2, M, N/16);;	  


	  const size_t numResBytes = sizeof (Type) * M * N;
	  const size_t numLhsBytes = sizeof (Type) * K * N;
	  const size_t numRhsBytes = sizeof (Type) * N * M;

	  ocl::Buffer bufRes( context_, numResBytes, ocl::Buffer::WriteOnly );
	  ocl::Buffer bufLhs( context_, numLhsBytes, ocl::Buffer::ReadOnly );
	  ocl::Buffer bufRhs( context_, numRhsBytes, ocl::Buffer::ReadOnly );

	  std::cout << "Running kernel with M=" << M << ", N=" << N << ", K=" << K << ", size[MB]=" 
	  << float(numLhsBytes + numRhsBytes + numResBytes)/float(1<<20) << std::endl;

	  Matrix lhs;
	  Matrix rhs;
	  if(testing_){
		  lhs = Ones ( N, K );
		  rhs = Ones ( K, M );
		  for ( size_t i = 0; i < M * N; ++i ) lhs[i] = i % N;
		  bufLhs.write( queue_, 0u, lhs.data(), numLhsBytes );
		  bufRhs.write( queue_, 0u, rhs.data(), numRhsBytes );
	  }


	  // Function which repeated iter_ times from the Passmanager.
	  auto lambda = [](ocl::Kernel& kernel, ocl::Queue& queue, ocl::Buffer& bufRes, const ocl::Buffer& bufLhs, 
	  	const ocl::Buffer& bufRhs, const int K, const int M, const int N)
	  {
		  kernel( queue, bufRes.id(), bufLhs.id(), bufRhs.id(), int(K), int(M), int(N) );
		  queue.finish();
	  };

	  auto t = this->call(std::bind(lambda, std::ref(*kernel_), std::ref(queue_), 
	  	std::ref(bufRes), std::cref(bufLhs), std::cref(bufRhs), std::cref(K), std::cref(M), std::cref(N)));


	  std::cout << "Zeit: "<< t << std::endl;
	  std::cout << "Flops: " << std::chrono::duration<double, std::ratio<1l> >(ops(dim))/t << std::endl;


	  if( testing_ )
	  {
		  Matrix res = Zeros( N, M );
		  bufRes.read( queue_, 0u, res.data(), numResBytes);

		  auto const ref  = lhs * rhs;
		  auto const diff = res - ref;
		  auto const iMax = std::max_element( diff.begin(), diff.end(), []( Type a, Type b ){ return std::fabs( a ) < std::fabs( b ); } );

//		  std::cout << "lhs = " << lhs << std::endl << "rhs = " << rhs << std::endl << "ref = " << ref << std::endl << "res = " << res << std::endl;

		  std::cout << "Maximal error: " << *iMax << std::endl;
		  if ( *iMax != 0 )
		  {
			  size_t const index = iMax - diff.begin();
			  std::cout << "ref[" << index << "] = " << ref[index] << " != res[" << index << "] = " << res[index] << std::endl;
		  }
	  }


	  program_.release();


	  return t;
}


/*! Operation count function of the Pass.
 *
 * Need to tell the PassManager how many operations are performed depending on the dimension.
 *
 * \param dim Dimension which is between the first and the last.
*/
template <class Type_,class Format_ , size_t W1, size_t W2>
double Stud4Pass1<Type_,Format_, W1,W2>::ops( utl::Dim const& dim )
{
	  size_t const M = dim[0];
	  size_t const N = dim[1];
	  size_t const K = dim[2];

	  return (M * K * (N + (N - 1u)));
}

#endif
