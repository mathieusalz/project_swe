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
#include <mpi.h>
#include <cstring>

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

  int psize, prank;
  MPI_Comm_size(MPI_COMM_WORLD, &psize);
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  std::shared_ptr<XDMFWriter> writer;

  if (prank == 0){
    if (output_n > 0)
    {
        writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, this->z_);
        writer->add_h(h0_, 0.0);
    }
  }

  double T = 0.0;

  std::vector<double> &h = h1_;
  std::vector<double> &hu = hu1_;
  std::vector<double> &hv = hv1_;

  std::vector<double> &h0 = h0_;
  std::vector<double> &hu0 = hu0_;
  std::vector<double> &hv0 = hv0_;

  if (prank == 0){
    std::cout << "Solving SWE..." << std::endl;
  }

  int base_rows = (ny_-2) / psize;
  int rem = (ny_-2) % psize;
  std::size_t p_local_rows = base_rows + (prank < rem ? 1 : 0);

  std::size_t start_row = 1 + prank * base_rows + (prank < rem ? prank : rem);

    std::vector<double> h_global, hu_global, hv_global;
  std::vector<int> recvcounts, displs;
  
  if (prank == 0) {
    h_global.resize(nx_ * ny_);
    hu_global.resize(nx_ * ny_);
    hv_global.resize(nx_ * ny_);
    recvcounts.resize(psize);
    displs.resize(psize);
    
    // Calculate receive counts and displacements for each process
    for (int p = 0; p < psize; ++p) {
      int p_base_rows = (ny_-2) / psize;
      int p_rem = (ny_-2) % psize;
      std::size_t p_rows = p_base_rows + (p < p_rem ? 1 : 0);
      recvcounts[p] = p_rows * nx_;
      
      std::size_t p_start_row = 1 + p * p_base_rows + (p < p_rem ? p : p_rem);
      displs[p] = p_start_row * nx_;
    }
  }

  std::size_t nt = 1;
  while (T < Tend)
  {
    const double dt = this->compute_time_step(h0, hu0, hv0, T, Tend, start_row, p_local_rows);

    const double T1 = T + dt;

    if (prank == 0){
      printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
      std::cout << (full_log ? "\n" : "\r") << std::flush;  
    }
    
    this->update_bcs(h0, hu0, hv0, h, hu, hv, prank, psize, start_row, p_local_rows);
    this->solve_step(dt, h0, hu0, hv0, h, hu, hv, start_row, p_local_rows);
    this->exchange_ghost_rows(h, hu, hv, prank, psize, start_row, p_local_rows);  

    if (output_n > 0 && nt % output_n == 0)
    {
      gather_solution_data(h, hu, hv, h_global, hu_global, hv_global, 
                          recvcounts, displs, p_local_rows);

      if (prank == 0){
        writer->add_h(h_global, T1);
      }
      
    }
    ++nt;

    
    // Swap the old and new solutions
    std::swap(h, h0);
    std::swap(hu, hu0);
    std::swap(hv, hv0);


    T = T1;
  }

  // Copying last computed values to h1_, hu1_, hv1_ (if needed)
  if (&h0 != &h1_)
  {
    h1_ = h0;
    hu1_ = hu0;
    hv1_ = hv0;
  }

  if (output_n > 0)
  {
    // Final gather operation
    gather_solution_data(h1_, hu1_, hv1_, h_global, hu_global, hv_global, 
                        recvcounts, displs, p_local_rows);
    
    if (prank == 0) {
      writer->add_h(h_global, T);
      // Update member variables to point to the complete global solution
      h1_ = std::move(h_global);
      hu1_ = std::move(hu_global);
      hv1_ = std::move(hv_global);
    }
  }
  else 
  {
    // Even if no output, we should gather the final solution for consistency
    gather_solution_data(h1_, hu1_, hv1_, h_global, hu_global, hv_global, 
                        recvcounts, displs, p_local_rows);
    
    if (prank == 0) {
      // Update member variables to point to the complete global solution
      h1_ = std::move(h_global);
      hu1_ = std::move(hu_global);
      hv1_ = std::move(hv_global);
    }
  }

  if (prank == 0) {
    std::cout << "Finished solving SWE." << std::endl;
  }
}


void SWESolver::exchange_ghost_rows(std::vector<double>& h,
                                    std::vector<double>& hu,
                                    std::vector<double>& hv,
                                    const int prank,
                                    const int psize,
                                    const std::size_t start_row,
                                    const std::size_t p_local_rows) const
{
    // Helper lambda to exchange a single field
    auto exchange_field = [&](std::vector<double>& field, int tag_send, int tag_recv) {
        MPI_Status status;
        std::vector<double> send_buf(nx_), recv_buf(nx_);
        
        // Exchange with upper neighbor (prank - 1)
        if (prank > 0) {
            std::size_t j_send = start_row;
            std::size_t j_recv = start_row - 1;
            
            // Copy row to send buffer
            for (std::size_t i = 0; i < nx_; ++i)
                send_buf[i] = at(field, i, j_send);
            
            // Send/receive
            MPI_Sendrecv(send_buf.data(), nx_, MPI_DOUBLE, prank - 1, tag_send,
                         recv_buf.data(), nx_, MPI_DOUBLE, prank - 1, tag_recv,
                         MPI_COMM_WORLD, &status);
            
            // Copy received data
            for (std::size_t i = 0; i < nx_; ++i)
                at(field, i, j_recv) = recv_buf[i];
        }
        
        // Exchange with lower neighbor (prank + 1)
        if (prank < psize - 1) {
            std::size_t j_send = start_row + p_local_rows - 1;
            std::size_t j_recv = start_row + p_local_rows;
            
            // Copy row to send buffer
            for (std::size_t i = 0; i < nx_; ++i)
                send_buf[i] = at(field, i, j_send);
            
            // Send/receive (note swapped tags for lower neighbor)
            MPI_Sendrecv(send_buf.data(), nx_, MPI_DOUBLE, prank + 1, tag_recv,
                         recv_buf.data(), nx_, MPI_DOUBLE, prank + 1, tag_send,
                         MPI_COMM_WORLD, &status);
            
            // Copy received data
            for (std::size_t i = 0; i < nx_; ++i)
                at(field, i, j_recv) = recv_buf[i];
        }
    };
    
    // Exchange all three fields
    exchange_field(h,  0, 1);  // h:  tags 0,1
    exchange_field(hu, 2, 3);  // hu: tags 2,3
    exchange_field(hv, 4, 5);  // hv: tags 4,5
}

void SWESolver::gather_solution_data(const std::vector<double>& h_local,
                                   const std::vector<double>& hu_local,
                                   const std::vector<double>& hv_local,
                                   std::vector<double>& h_global,
                                   std::vector<double>& hu_global,
                                   std::vector<double>& hv_global,
                                   const std::vector<int>& recvcounts,
                                   const std::vector<int>& displs,
                                   std::size_t p_local_rows
                                   ) const
{
  // Extract local interior data (excluding ghost cells)
  std::vector<double> h_send_buf(p_local_rows * nx_);
  std::vector<double> hu_send_buf(p_local_rows * nx_);
  std::vector<double> hv_send_buf(p_local_rows * nx_);
  
  for (std::size_t i = 0; i < p_local_rows; ++i) {
    for (std::size_t j = 0; j < nx_; ++j) {
      std::size_t local_idx = (i + 1) * nx_ + j;  // +1 to skip ghost row
      std::size_t send_idx = i * nx_ + j;
      
      h_send_buf[send_idx] = h_local[local_idx];
      hu_send_buf[send_idx] = hu_local[local_idx];
      hv_send_buf[send_idx] = hv_local[local_idx];
    }
  }
  
  // Gather data using MPI_Gatherv
  MPI_Gatherv(h_send_buf.data(), p_local_rows * nx_, MPI_DOUBLE,
              h_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
              
  MPI_Gatherv(hu_send_buf.data(), p_local_rows * nx_, MPI_DOUBLE,
              hu_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
              
  MPI_Gatherv(hv_send_buf.data(), p_local_rows * nx_, MPI_DOUBLE,
              hv_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
}

double
SWESolver::compute_time_step(const std::vector<double> &h,
                             const std::vector<double> &hu,
                             const std::vector<double> &hv,
                             const double T,
                             const double Tend,
                             const std::size_t start_row,
                             const std::size_t p_local_rows) const
{
  double max_nu_sqr = 0.0;
  double au{0.0};
  double av{0.0};
  for (std::size_t j = start_row; j < start_row + p_local_rows; ++j)
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
  double global_max_nu_sqr;
  MPI_Allreduce(&max_nu_sqr, &global_max_nu_sqr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  
  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;
  double dt = std::min(dx, dy) / (sqrt(2.0 * global_max_nu_sqr));
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
                      std::vector<double> &hv,
                      const std::size_t start_row,
                      const std::size_t p_local_rows
                      ) const
{
  for (std::size_t j = start_row; j < start_row + p_local_rows; ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      this->compute_kernel(i, j, dt, h0, hu0, hv0, h, hu, hv);
    }
  }
}

void
SWESolver::update_bcs(const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv,
                      const int prank,
                      const int psize,
                      std::size_t start_row,
                      std::size_t p_local_rows) const
{
  const double coef = this->reflective_ ? -1.0 : 1.0;

  // Top and bottom boundaries.
  if (prank == 0){
    for (std::size_t i = 0; i < nx_; ++i)
    {
        at(h, i, 0) = at(h0, i, 1);
        at(hu, i, 0) = at(hu0, i, 1);
        at(hv, i, 0) = coef * at(hv0, i, 1);
    }
  }

  if (prank == psize -1){
    for (std::size_t i = 0; i < nx_; ++i)
    {
        at(h, i, ny_ - 1) = at(h0, i, ny_ - 2);
        at(hu, i, ny_ - 1) = at(hu0, i, ny_ - 2);
        at(hv, i, ny_ - 1) = coef * at(hv0, i, ny_ - 2);
    }
  }

    std::size_t bc_start_row = start_row;
    std::size_t bc_rows = p_local_rows;
    
    if (prank == 0) {
        bc_start_row = 0;
        bc_rows = p_local_rows + 1;
    } else if (prank == psize - 1) {
        bc_rows = p_local_rows + 1;
    }

    // Left and right boundaries
    for (std::size_t j = bc_start_row; j < bc_start_row + bc_rows; ++j) {
        at(h, 0, j) = at(h0, 1, j);
        at(h, nx_ - 1, j) = at(h0, nx_ - 2, j);

        at(hu, 0, j) = coef * at(hu0, 1, j);
        at(hu, nx_ - 1, j) = coef * at(hu0, nx_ - 2, j);

        at(hv, 0, j) = at(hv0, 1, j);
        at(hv, nx_ - 1, j) = at(hv0, nx_ - 2, j);
    }
};
