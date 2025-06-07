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

__global__ void kernel_timeStep(const double* h,
                                  const double* hu,
                                  const double* hv,
                                  const int T,
                                  const int Tend,
                                  const std::size_t nx,
                                  const std::size_t ny,
                                  const double g,
                                  double* block_results,
                                  const double size_x_,
                                  const double size_y_);

__global__ void kernel_solveStep(const double* h0,
                                const double* hu0,
                                const double* hv0,
                                const double dt,
                                const std::size_t nx,
                                const std::size_t ny,
                                double* h,
                                double* hu,
                                double* hv,
                                const double* zdx,
                                const double* zdy,
                                const double size_x, 
                                const double size_y,
                                const double g);

__global__ void kernel_updateBCs(const double* h0, const double* hu0, const double* hv0,
                                  double* h, double* hu, double* hv,
                                  int nx, int ny, double coef);

__global__ void reduce_max_kernel(double* input, double* output, int n);

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

    const double coef = this->reflective_ ? -1.0 : 1.0;
    int num_blocks = grid_inner.x * grid_inner.y;
    
    double *d_block_results, *d_final_result, *d_temp_results; 
    cudaMalloc(&d_block_results, num_blocks * sizeof(double));
    cudaMalloc(&d_temp_results, num_blocks * sizeof(double));
    cudaMalloc(&d_final_result, sizeof(double));

    const int total_boundary_elements = 2 * nx_ + 2 * ny_;
    const int TBP_updateBCs = 256;
    const int num_blocks_updateBCs = (total_boundary_elements + TBP_updateBCs - 1) / TBP_updateBCs;

    std::size_t nt = 1;
    while (T < Tend)
    {
        size_t shared_mem_size = TPB.x * TPB.y * sizeof(double);
        kernel_timeStep<<<grid_inner, TPB, shared_mem_size>>>(
            d_h0, d_hu0, d_hv0, T, Tend, nx_, ny_, g, d_block_results, size_x_, size_y_);

        double *tobe_reduced = d_block_results;
        double *reduced = d_temp_results;
        int current_size = num_blocks;
        
        while (current_size > 1) {
            int threads_per_block = 256;  // Must be power of 2
            int blocks_needed = (current_size + threads_per_block - 1) / threads_per_block;
            
            reduce_max_kernel<<<blocks_needed, threads_per_block, 
                              threads_per_block * sizeof(double)>>>(
                tobe_reduced, reduced, current_size);
            
            // Swap pointers for next iteration
            std::swap(tobe_reduced, reduced);
            current_size = blocks_needed;
        }
        
        double max_nu_sqr;
        cudaMemcpy(&max_nu_sqr, tobe_reduced, sizeof(double), cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // cudaMemcpy(block_results, d_block_results, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
        // double max_nu_sqr = 0.0;
        // for (int i = 0; i < num_blocks; i++) {
        //     max_nu_sqr = std::max(max_nu_sqr, block_results[i]);
        // }
      
        printf("Max nu^2: %.4e\n", max_nu_sqr);
        const double dx = size_x_ / nx_;
        const double dy = size_y_ / ny_;
        double dt = min(dx, dy) / sqrt(2.0 * max_nu_sqr);
        dt = min(dt, Tend - T);

        const double T1 = T + dt;

        printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
        std::cout << (full_log ? "\n" : "\r") << std::flush;

        kernel_updateBCs<<<num_blocks_updateBCs, TBP_updateBCs>>>(d_h0, d_hu0, d_hv0, d_h, d_hu, d_hv, nx_, ny_, coef);
        cudaDeviceSynchronize();
        kernel_solveStep<<<grid_inner, TPB>>>(d_h0, d_hu0, d_hv0, dt, nx_, ny_, d_h, d_hu, d_hv, d_zdx, d_zdy, size_x_, size_y_, g);
        cudaDeviceSynchronize();

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

__global__ void kernel_timeStep(const double* h,
                              const double* hu,
                              const double* hv,
                              const int T,
                              const int Tend,
                              const std::size_t nx,
                              const std::size_t ny,
                              const double g,
                              double* block_results,
                              const double size_x_,
                              const double size_y_)
{
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    extern __shared__ double sdata[];
    
    double max_nu_sqr = 0.0;
    
    // Check if thread is within the computation domain (matches your loop bounds)
    if (thread_x >= 1 && thread_x < nx - 1 && thread_y >= 1 && thread_y < ny - 1) {
        
        int idx = thread_x + thread_y * nx;
        
        double hu_val = hu[idx];
        double hv_val = hv[idx];
        double h_val = h[idx];

        const double nu_u = fabs(hu_val) / h_val + sqrt(g * h_val);
        const double nu_v = fabs(hv_val) / h_val + sqrt(g * h_val);
        
        max_nu_sqr = fmax(max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
    } 
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    sdata[tid] = max_nu_sqr;
    __syncthreads();
    
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);  
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_results[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0];
    }
}

__global__ void reduce_max_kernel(double* input, double* output, int n) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory (handle out-of-bounds with -infinity for max)
    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    __syncthreads();
    
    // Standard reduction loop
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
__global__ void kernel_solveStep(const double* h0,
                                const double* hu0,
                                const double* hv0,
                                const double dt,
                                const std::size_t nx,
                                const std::size_t ny,
                                double* h,
                                double* hu,
                                double* hv,
                                const double* zdx,
                                const double* zdy,
                                const double size_x, 
                                const double size_y,
                                const double g)       
{
  int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (thread_x < 1 || thread_x >= nx - 1 || thread_y < 1 || thread_y >= ny - 1) {
    return;
  }

  const double dx = size_x / nx;
  const double dy = size_y / ny;
  const double C1x = 0.5 * dt / dx;
  const double C1y = 0.5 * dt / dy;
  const double C2 = dt * g;
  const double C3 = 0.5 * g;

  int idx = thread_x + thread_y * nx; 
  int idx_l = (thread_x - 1) + thread_y * nx;
  int idx_r = (thread_x + 1) + thread_y * nx;
  int idx_t = thread_x + (thread_y + 1) * nx;
  int idx_b = thread_x + (thread_y - 1) * nx;
  
  double hij = 0.25 * (h0[idx_b] + h0[idx_t] + h0[idx_r] + h0[idx_l])       
               + C1x * (hu0[idx_l] - hu0[idx_r])     
               + C1y * (hv0[idx_b] - hv0[idx_t]);    
  
  if (hij < 0.0) {
    hij = 1.0e-5;
  }

  h[idx] = hij;

  if (hij > 0.0001) {
    hu[idx] = 
      0.25 * (hu0[idx_b] + hu0[idx_t] + hu0[idx_l]  + hu0[idx_r])               
      - C2 * hij * zdx[idx]                                      
      + C1x * (hu0[idx_l] * hu0[idx_l] / h0[idx_l] + C3 * h0[idx_l] * h0[idx_l]
             - hu0[idx_r] * hu0[idx_r] / h0[idx_r] - C3 * h0[idx_r] * h0[idx_r])
      + C1y * (hu0[idx_b] * hv0[idx_b] / h0[idx_b]
             - hu0[idx_t] * hv0[idx_t] / h0[idx_t]);

    hv[idx] = 
      0.25 * (hv0[idx_b] + hv0[idx_t] + hv0[idx_l] + hv0[idx_r])              
      - C2 * hij * zdy[idx]                                      
      + C1x * (hu0[idx_l] * hv0[idx_l] / h0[idx_l]
             - hu0[idx_r] * hv0[idx_r] / h0[idx_r])
      + C1y * (hv0[idx_b] * hv0[idx_b] / h0[idx_b] + C3 * h0[idx_b] * h0[idx_b]
             - hv0[idx_t] * hv0[idx_t] / h0[idx_t] - C3 * h0[idx_t] * h0[idx_t]);
  } else {
    hu[idx] = 0.0;
    hv[idx] = 0.0;
  }
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
        // Top boundary 
        int i = idx;
        int src_idx = i + 1;        
        int dst_idx = i + 0;        
        
        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = hu0[src_idx];      
        hv[dst_idx] = coef * hv0[src_idx];
        
    } else if (idx < 2 * nx) {
        // Bottom boundary 
        int i = idx - nx;
        int src_idx = i + (ny - 2) * nx;  
        int dst_idx = i + (ny - 1) * nx;  
        
        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = hu0[src_idx];       
        hv[dst_idx] = coef * hv0[src_idx]; 
        
    } else if (idx < 2 * nx + ny) {
        // Left boundary 
        int j = idx - 2 * nx;
        int src_idx = 1 + j * nx;         
        int dst_idx = 0 + j * nx;         
        
        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = coef * hu0[src_idx]; 
        hv[dst_idx] = hv0[src_idx];        
        
    } else {
        // Right boundary 
        int j = idx - 2 * nx - ny;
        int src_idx = (nx - 2) + j * nx;  
        int dst_idx = (nx - 1) + j * nx;  

        h[dst_idx] = h0[src_idx];
        hu[dst_idx] = coef * hu0[src_idx]; 
        hv[dst_idx] = hv0[src_idx];
    }
}
;
