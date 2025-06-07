#include "swe.hh"
#include "xdmf_writer.hh"

#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstdio>
#include <cmath>
#include <memory>
#include <cuda_runtime.h>


namespace
{

void
read_2d_array_from_DF5(const std::string &filename,
                       const std::string &dataset_name,
                       std::vector<double> &data,
                       std::size_t &nx,
                       std::size_t &ny)
{
  hid_t file_id, dataset_id, dataspace_id;
  hsize_t dims[2];
  herr_t status;

  // Open the HDF5 file
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error opening HDF5 file: " << filename << std::endl;
    return;
  }

  // Open the dataset
  dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error opening dataset: " << dataset_name << std::endl;
    H5Fclose(file_id);
    return;
  }

  // Get the dataspace
  dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0)
  {
    std::cerr << "Error getting dataspace" << std::endl;
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }

  // Get the dimensions of the dataset
  status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
  if (status < 0)
  {
    std::cerr << "Error getting dimensions" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }
  nx = dims[0];
  ny = dims[1];

  // Resize the data vector
  data.resize(nx * ny);

  // Read the data
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  if (status < 0)
  {
    std::cerr << "Error reading data" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    data.clear();
    return;
  }

  // Close resources
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  // std::cout << "Successfully read 2D array from HDF5 file: " << filename << ", dataset: " << dataset_name <<
  // std::endl;
}

} // namespace

SWESolver::SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny) :
  nx_(nx), ny_(ny), size_x_(500.0), size_y_(500.0)
{
  assert(test_case_id == 1 || test_case_id == 2);
  if (test_case_id == 1)
  {
    this->reflective_ = true;
    this->init_gaussian();
  }
  else if (test_case_id == 2)
  {
    this->reflective_ = false;
    this->init_dummy_tsunami();
  }
  else
  {
    assert(false);
  }
}

SWESolver::SWESolver(const std::string &h5_file, const double size_x, const double size_y) :
  size_x_(size_x), size_y_(size_y), reflective_(false)
{
  this->init_from_HDF5_file(h5_file);
}

void
SWESolver::init_from_HDF5_file(const std::string &h5_file)
{
  read_2d_array_from_DF5(h5_file, "h0", this->h0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hu0", this->hu0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hv0", this->hv0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "topography", this->z_, this->nx_, this->ny_);

  this->h1_.resize(this->h0_.size(), 0.0);
  this->hu1_.resize(this->hu0_.size(), 0.0);
  this->hv1_.resize(this->hv0_.size(), 0.0);

  this->init_dx_dy();
}

void
SWESolver::init_gaussian()
{
  hu0_.resize(nx_ * ny_, 0.0);
  hv0_.resize(nx_ * ny_, 0.0);
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);

  h0_.clear();
  h0_.reserve(nx_ * ny_);

  h1_.resize(nx_ * ny_);
  hu1_.resize(nx_ * ny_);
  hv1_.resize(nx_ * ny_);

  const double x0_0 = size_x_ / 4.0;
  const double y0_0 = size_y_ / 3.0;
  const double x0_1 = size_x_ / 2.0;
  const double y0_1 = 0.75 * size_y_;

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx * (static_cast<double>(i) + 0.5);
      const double y = dy * (static_cast<double>(j) + 0.5);
      const double gauss_0 = 10.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
      const double gauss_1 = 10.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);

      h0_.push_back(10.0 + gauss_0 + gauss_1);
    }
  }

  z_.resize(this->h0_.size());
  std::fill(z_.begin(), z_.end(), 0.0);

  this->init_dx_dy();
}

void
SWESolver::init_dummy_tsunami()
{
  hu0_.resize(nx_ * ny_);
  hv0_.resize(nx_ * ny_);
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);

  h1_.resize(nx_ * ny_);
  hu1_.resize(nx_ * ny_);
  hv1_.resize(nx_ * ny_);
  std::fill(h1_.begin(), h1_.end(), 0.0);
  std::fill(hu1_.begin(), hu1_.end(), 0.0);
  std::fill(hv1_.begin(), hv1_.end(), 0.0);

  const double x0_0 = 0.6 * size_x_;
  const double y0_0 = 0.6 * size_y_;
  const double x0_1 = 0.4 * size_x_;
  const double y0_1 = 0.4 * size_y_;
  const double x0_2 = 0.7 * size_x_;
  const double y0_2 = 0.3 * size_y_;

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  // Creating topography and initial water height
  z_.resize(nx_ * ny_);
  h0_.resize(nx_ * ny_);
  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx * (static_cast<double>(i) + 0.5);
      const double y = dy * (static_cast<double>(j) + 0.5);

      const double gauss_0 = 2.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
      const double gauss_1 = 3.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
      const double gauss_2 = 5.0 * std::exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);

      const double z = -1.0 + gauss_0 + gauss_1;
      at(z_, i, j) = z;

      double h0 = z < 0.0 ? -z + gauss_2 : 0.00001;
      at(h0_, i, j) = h0;
    }
  }
  this->init_dx_dy();
}

void
SWESolver::init_dummy_slope()
{
  hu0_.resize(nx_ * ny_);
  hv0_.resize(nx_ * ny_);
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);

  h1_.resize(nx_ * ny_);
  hu1_.resize(nx_ * ny_);
  hv1_.resize(nx_ * ny_);
  std::fill(h1_.begin(), h1_.end(), 0.0);
  std::fill(hu1_.begin(), hu1_.end(), 0.0);
  std::fill(hv1_.begin(), hv1_.end(), 0.0);

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  const double dz = 10.0;

  // Creating topography and initial water height
  z_.resize(nx_ * ny_);
  h0_.resize(nx_ * ny_);
  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx * (static_cast<double>(i) + 0.5);
      const double y = dy * (static_cast<double>(j) + 0.5);
      static_cast<void>(y);

      const double z = -10.0 - 0.5 * dz + dz / size_x_ * x;
      at(z_, i, j) = z;

      double h0 = z < 0.0 ? -z : 0.00001;
      at(h0_, i, j) = h0;
    }
  }
  this->init_dx_dy();
}

void
SWESolver::init_dx_dy()
{
  zdx_.resize(this->z_.size(), 0.0);
  zdy_.resize(this->z_.size(), 0.0);

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;
  for (std::size_t j = 1; j < ny_ - 1; ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      at(this->zdx_, i, j) = 0.5 * (at(this->z_, i + 1, j) - at(this->z_, i - 1, j)) / dx;
      at(this->zdy_, i, j) = 0.5 * (at(this->z_, i, j + 1) - at(this->z_, i, j - 1)) / dy;
    }
  }
}

    void
    SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix)
    {
    std::shared_ptr<XDMFWriter> writer;
    if (output_n > 0)
    {
    writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, this->z_);
    writer->add_h(h0_, 0.0);
    }

    double T = 0.0;

    std::vector<double> &h = h1_;
    std::vector<double> &hu = hu1_;
    std::vector<double> &hv = hv1_;
    std::vector<double> &h0 = h0_;
    std::vector<double> &hu0 = hu0_;
    std::vector<double> &hv0 = hv0_;

    double* d_h = nullptr;
    double* d_hu = nullptr;
    double* d_hv = nullptr;
    double* d_h0 = nullptr;
    double* d_hu0 = nullptr;
    double* d_hv0 = nullptr;
    double* d_zdx = nullptr;
    double* d_zdy = nullptr;

    cudaMalloc(&d_h, h.size() * sizeof(double));
    cudaMalloc(&d_hu, hu.size() * sizeof(double));
    cudaMalloc(&d_hv, hv.size() * sizeof(double));
    cudaMalloc(&d_h0, h0.size() * sizeof(double));
    cudaMalloc(&d_hu0, hu0.size() * sizeof(double));
    cudaMalloc(&d_hv0, hv0.size() * sizeof(double));
    cudaMalloc(&d_zdx, zdx_.size() * sizeof(double));
    cudaMalloc(&d_zdy, zdy_.size() * sizeof(double));

    cudaMemcpy(d_h0, h0.data(), h0.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hu0, hu0.data(), hu0.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hv0, hv0.data(), hv0.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zdx, zdx_.data(), zdx_.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zdy, zdy_.data(), zdy_.size() * sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "Solving SWE..." << std::endl;

    dim3 TPB(32, 32); 
    dim3 grid_inner(((nx_-2) + TPB.x - 1) / TPB.x, ((ny_-2) + TPB.y - 1) / TPB.y);
    dim3 grid_total((nx_ + TPB.x - 1) / TPB.x, (ny_ + TPB.y - 1) / TPB.y);

    dim3 grid_total = dim3((nx_ + TPB.x - 1) / TPB.x, (ny_ + TPB.y - 1) / TPB.y);

    const double coef = this->reflective_ ? -1.0 : 1.0;
    const int total_boundary_elements = 2 * nx_ + 2 * ny_;

    double *d_block_results, *d_final_result;
    cudaMalloc(&d_block_results, num_blocks * sizeof(double));
    cudaMalloc(&d_final_result, sizeof(double));

    std::size_t nt = 1;
    while (T < Tend)
    {
        size_t shared_mem_size = TPB.x * TPB.y * sizeof(double);
        kernel_timeStep<<<grid_inner, TPB, shared_mem_size>>>(
            d_h, d_hu, d_hv, d_block_results, nx_, ny_, g, dx, dy
        );

        int final_block_size = min(num_blocks, 256);
        int final_grid_size = 1;

        reduce_min_kernel<<<final_grid_size, final_block_size, final_block_size * sizeof(double)>>>(d_block_results, d_final_result, num_blocks);

        double dt;
        cudaMemcpy(&dt, d_final_result, sizeof(double), cudaMemcpyDeviceToHost);

        const double T1 = T + dt;

        printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
        std::cout << (full_log ? "\n" : "\r") << std::flush;

        kernel_updateBCs<<<1, total_boundary_elements>>>(d_h0, d_hu0, d_hv0, d_h, d_hu, d_hv, nx_, ny_, coef);

        kernel_solveStep<<<grid_inner, TPB>>>(d_h0, d_hu0, d_hv0, dt, nx_, ny_, d_h, d_hu, d_hv, d_zdx, d_zdy);

        if (output_n > 0 && nt % output_n == 0)
        {
        cudaMemcpy(h.data(), d_h, h0.size() * sizeof(double), cudaMemcpyDeviceToHost);
        writer->add_h(h, T1);
        }
        ++nt;

        // Swap the old and new solutions

        std::swap(d_h0, d_h);
        std::swap(d_hu0, d_hu);
        std::swap(d_hv0, d_hv);

        T = T1;
    }

  cudaMemcpy(h0.data(), d_h0, h0.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hu0.data(), d_hu0, h0.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hv0.data(), d_hv0, h0.size() * sizeof(double), cudaMemcpyDeviceToHost);

  // Copying last computed values to h1_, hu1_, hv1_ (if needed)
  if (&h0 != &h1_)
  {
    h1_ = h0;
    hu1_ = hu0;
    hv1_ = hv0;
  }

  cudaFree(d_h);
  cudaFree(d_hu);
  cudaFree(d_hv);
  cudaFree(d_h0);
  cudaFree(d_hu0);
  cudaFree(d_hv0);
  cudaFree(d_zdx);
  cudaFree(d_zdy);
  cudaFree(d_block_results);
  cudaFree(d_final_result);

  if (output_n > 0)
  {
    writer->add_h(h1_, T);
  }

  std::cout << "Finished solving SWE." << std::endl;
}

__global__ double kernel_timeStep(const double* h,
                                  const double* hu,
                                  const double* hv,
                                  const int T,
                                  const int Tend,
                                  const std::size_t nx,
                                  const std::size_t ny,
                                  const double g,
                                  double* block_results)
{

  int threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int threadIdy = blockIdx.y * blockDim.y + threadIdx.y;


  if (threadIdx >= nx || threadIdy >= ny || threadIdx == 0 || threadIdy == 0 || threadIdx == nx - 1 || threadIdy == ny - 1)
  {
    return 0.0; // Out of bounds
  }

  extern __shared__ double sdata[];

  double max_nu_sqr = 0.0;
  double au{0.0};
  double av{0.0};

  au = fmax(au, fabs(hu[threadIdx + threadIdy * nx]));
  av = fmax(av, fabs(hv[threadIdx + threadIdy * nx]));
  const double nu_u = fabs(hu[threadIdx + threadIdy * nx]) / h[threadIdx + threadIdy * nx] + sqrt(g * h[threadIdx + threadIdy * nx]);
  const double nu_v = fabs(hv[threadIdx + threadIdy * nx]) / h[threadIdx + threadIdy * nx] + sqrt(g * h[threadIdx + threadIdy * nx]);
  max_nu_sqr = fmax(max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
    
  const double dx = size_x_ / static_cast<double>(nx);
  const double dy = size_y_ / static_cast<double>(ny);

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
    sdata[tid] = local_dt;
    __syncthreads();

    // Block-level reduction to find minimum
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        block_results[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0];
    }
}

__global__ void reduce_min_kernel(double* block_results, double* final_result, int num_blocks)
{
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < num_blocks) {
        sdata[tid] = block_results[idx];
    } else {
        sdata[tid] = DBL_MAX;
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        *final_result = sdata[0];
    }
}

double
SWESolver::compute_time_step(const std::vector<double> &h,
                             const std::vector<double> &hu,
                             const std::vector<double> &hv,
                             const double T,
                             const double Tend) const
{
  double max_nu_sqr = 0.0;
  double au{0.0};
  double av{0.0};
  for (std::size_t j = 1; j < ny_ - 1; ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      au = std::max(au, std::fabs(at(hu, i, j)));
      av = std::max(av, std::fabs(at(hv, i, j)));
      const double nu_u = std::fabs(at(hu, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
      const double nu_v = std::fabs(at(hv, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
      max_nu_sqr = std::max(max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
    }
  }

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;
  double dt = std::min(dx, dy) / (sqrt(2.0 * max_nu_sqr));
  return std::min(dt, Tend - T);
}

void
SWESolver::compute_kernel(const std::size_t i,
                          const std::size_t j,
                          const double dt,
                          const std::vector<double> &h0,
                          const std::vector<double> &hu0,
                          const std::vector<double> &hv0,
                          std::vector<double> &h,
                          std::vector<double> &hu,
                          std::vector<double> &hv) const
{
  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;
  const double C1x = 0.5 * dt / dx;
  const double C1y = 0.5 * dt / dy;
  const double C2 = dt * g;
  constexpr double C3 = 0.5 * g;

  double hij = 0.25 * (at(h0, i, j - 1) + at(h0, i, j + 1) + at(h0, i - 1, j) + at(h0, i + 1, j))
               + C1x * (at(hu0, i - 1, j) - at(hu0, i + 1, j)) + C1y * (at(hv0, i, j - 1) - at(hv0, i, j + 1));
  if (hij < 0.0)
  {
    hij = 1.0e-5;
  }

  at(h, i, j) = hij;

  if (hij > 0.0001)
  {
    at(hu, i, j) =
      0.25 * (at(hu0, i, j - 1) + at(hu0, i, j + 1) + at(hu0, i - 1, j) + at(hu0, i + 1, j)) - C2 * hij * at(zdx_, i, j)
      + C1x
          * (at(hu0, i - 1, j) * at(hu0, i - 1, j) / at(h0, i - 1, j) + C3 * at(h0, i - 1, j) * at(h0, i - 1, j)
             - at(hu0, i + 1, j) * at(hu0, i + 1, j) / at(h0, i + 1, j) - C3 * at(h0, i + 1, j) * at(h0, i + 1, j))
      + C1y
          * (at(hu0, i, j - 1) * at(hv0, i, j - 1) / at(h0, i, j - 1)
             - at(hu0, i, j + 1) * at(hv0, i, j + 1) / at(h0, i, j + 1));

    at(hv, i, j) =
      0.25 * (at(hv0, i, j - 1) + at(hv0, i, j + 1) + at(hv0, i - 1, j) + at(hv0, i + 1, j)) - C2 * hij * at(zdy_, i, j)
      + C1x
          * (at(hu0, i - 1, j) * at(hv0, i - 1, j) / at(h0, i - 1, j)
             - at(hu0, i + 1, j) * at(hv0, i + 1, j) / at(h0, i + 1, j))
      + C1y
          * (at(hv0, i, j - 1) * at(hv0, i, j - 1) / at(h0, i, j - 1) + C3 * at(h0, i, j - 1) * at(h0, i, j - 1)
             - at(hv0, i, j + 1) * at(hv0, i, j + 1) / at(h0, i, j + 1) - C3 * at(h0, i, j + 1) * at(h0, i, j + 1));
  }
  else
  {
    at(hu, i, j) = 0.0;
    at(hv, i, j) = 0.0;
  }

}

void
SWESolver::solve_step(const double dt,
                      const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const
{
  for (std::size_t j = 1; j < ny_ - 1; ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      this->compute_kernel(i, j, dt, h0, hu0, hv0, h, hu, hv);
    }
  }
}

__global__ kernel_solveStep(const double* h0,
                                  const double* hu0,
                                  const double* hv0,
                                  const double dt,
                                  const std::size_t nx,
                                  const std::size_t ny,
                                  double* h,
                                  double* hu,
                                  double* hv,
                                  const double* zdx,
                                  const double* zdy)
{
  int threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int threadIdy = blockIdx.y * blockDim.y + threadIdx.y;

  if (threadIdx >= nx || threadIdy >= ny || threadIdx == 0 || threadIdy == 0 || threadIdx == nx - 1 || threadIdy == ny - 1)
  {
    return 0.0; // Out of bounds
  }

  // Call the compute_kernel function for each thread
  compute_kernel(threadIdx, threadIdy, dt, h0, hu0, hv0, h, hu, hv, zdx, zdy);
}


__global__ void kernel_updateBCs(const double* h0, const double* hu0, const double* hv0,
                                  double* h, double* hu, double* hv,
                                  int nx, int ny, double coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total boundary elements: 2*nx (top/bottom) + 2*ny (left/right)
    int total_boundary_elements = 2 * nx + 2 * ny;
    
    if (idx >= total_boundary_elements) return;
    
    // Determine which boundary this thread handles
    if (idx < nx) {
        // Top boundary (j = 0)
        int i = idx;
        int src_idx = i * ny + 1;        // (i, 1)
        int dst_idx = i * ny + 0;        // (i, 0)
        
        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = hu0[src_idx];
        hv[dst_idx] = coef * hv0[src_idx];
        
    } else if (idx < 2 * nx) {
        // Bottom boundary (j = ny-1)
        int i = idx - nx;
        int src_idx = i * ny + (ny - 2);  // (i, ny-2)
        int dst_idx = i * ny + (ny - 1);  // (i, ny-1)
        
        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = hu0[src_idx];
        hv[dst_idx] = coef * hv0[src_idx];
        
    } else if (idx < 2 * nx + ny) {
        // Left boundary (i = 0)
        int j = idx - 2 * nx;
        int src_idx = 1 * ny + j;         // (1, j)
        int dst_idx = 0 * ny + j;         // (0, j)
        
        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = hu0[src_idx];
        hv[dst_idx] = coef * hv0[src_idx];
        
    } else {
        // Right boundary (i = nx-1)
        int j = idx - 2 * nx - ny;
        int src_idx = (nx - 2) * ny + j;  // (nx-2, j)
        int dst_idx = (nx - 1) * ny + j;  // (nx-1, j)
        
        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = hu0[src_idx];
        hv[dst_idx] = coef * hv0[src_idx];
    }
}


void
SWESolver::update_bcs(const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const
{
  const double coef = this->reflective_ ? -1.0 : 1.0;

  // Top and bottom boundaries.
  for (std::size_t i = 0; i < nx_; ++i)
  {
    at(h, i, 0) = at(h0, i, 1);
    at(h, i, ny_ - 1) = at(h0, i, ny_ - 2);

    at(hu, i, 0) = at(hu0, i, 1);
    at(hu, i, ny_ - 1) = at(hu0, i, ny_ - 2);

    at(hv, i, 0) = coef * at(hv0, i, 1);
    at(hv, i, ny_ - 1) = coef * at(hv0, i, ny_ - 2);
  }

  // Left and right boundaries.
  for (std::size_t j = 0; j < ny_; ++j)
  {
    at(h, 0, j) = at(h0, 1, j);
    at(h, nx_ - 1, j) = at(h0, nx_ - 2, j);

    at(hu, 0, j) = coef * at(hu0, 1, j);
    at(hu, nx_ - 1, j) = coef * at(hu0, nx_ - 2, j);

    at(hv, 0, j) = at(hv0, 1, j);
    at(hv, nx_ - 1, j) = at(hv0, nx_ - 2, j);
  }
};
