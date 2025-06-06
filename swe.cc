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

  double T = 0.0;

  std::shared_ptr<XDMFWriter> writer;
  std::vector<double>* h_global = nullptr;
  std::vector<double>* hu_global = nullptr;
  std::vector<double>* hv_global = nullptr;

  std::vector<double>* h0_global = nullptr;
  std::vector<double>* hu0_global = nullptr;
  std::vector<double>* hv0_global = nullptr;

  if (prank == 0) {
    std::shared_ptr<XDMFWriter> writer;
    if (output_n > 0)
    {
      writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, this->z_);
      writer->add_h(h0_, 0.0);
    }

    h_global = &h1_;
    hu_global = &hu1_;
    hv_global = &hv1_;

    h0_global = &h0_;
    hu0_global = &hu0_;
    hv0_global = &hv0_;

    std::cout << "Solving SWE..." << std::endl;
  }

  int base_rows = ny_ / psize;
  int rem = ny_ % psize;
  int p_local_rows = base_rows + (prank < rem ? 1 : 0);

  int p_extra_top = (prank == 0) ? 0 : 1;
  int p_extra_bottom = (prank == psize - 1) ? 0 : 1;

  int p_rows_to_copy = p_local_rows + p_extra_top + p_extra_bottom;
  int p_count = p_rows_to_copy * nx_;

  // Allocate memory for the local solution vectors
  double* h_local = (double*)malloc(p_count * sizeof(double));
  double* hu_local = (double*)malloc(p_count * sizeof(double));
  double* hv_local = (double*)malloc(p_count * sizeof(double));
  double* h0_local = (double*)malloc(p_count * sizeof(double));
  double* hu0_local = (double*)malloc(p_count * sizeof(double));
  double* hv0_local = (double*)malloc(p_count * sizeof(double));

  this->distribute_initial_data(
  h0_local, hu0_local, hv0_local,
  *h0_global, *hu0_global, *hv0_global,
  nx_, base_rows, rem,
  prank, psize, p_count
  );

  std::size_t nt = 1;
  while (T < Tend)
  {
    const double local_dt = this->compute_time_step(h0_local, hu0_local, hv0_local, T, Tend, p_local_rows, prank);

    double global_dt;
    MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    
    const double T1 = T + global_dt;

    printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, global_dt * 3600, 100 * T1 / Tend);
    std::cout << (full_log ? "\n" : "\r") << std::flush;

    this->update_bcs(h0_local, hu0_local, hv0_local, h_local, hu_local, hv_local, p_local_rows, prank, psize);

    this->solve_step(global_dt, h0_local, hu0_local, hv0_local, h_local, hu_local, hv_local, p_local_rows, prank);

    this->exchange_ghost_rows(h0_local,  nx_, p_local_rows, prank, psize, MPI_COMM_WORLD);
    this->exchange_ghost_rows(hu0_local, nx_, p_local_rows, prank, psize, MPI_COMM_WORLD);
    this->exchange_ghost_rows(hv0_local, nx_, p_local_rows, prank, psize, MPI_COMM_WORLD);

    if (output_n > 0 && nt % output_n == 0)
    {

      this->gather_fields_to_root(h_local, hu_local, hv_local,
                      h_global->data(), hu_global->data(), hv_global->data(),
                      nx_, psize, base_rows, rem,
                      p_local_rows, p_extra_top);
      if (prank == 0){
        writer->add_h(*h_global, T1);
      }
    }
    ++nt;

    // Swap the old and new solutions
    std::swap(h_local, h0_local);
    std::swap(hu_local, hu0_local);
    std::swap(hv_local, hv0_local);

    T = T1;
  }

  this->gather_fields_to_root(h_local, hu_local, hv_local,
                      h_global->data(), hu_global->data(), hv_global->data(),
                      nx_, psize, base_rows, rem,
                      p_local_rows, p_extra_top);

  free(h_local);
  free(hu_local);
  free(hv_local);
  free(h0_local);
  free(hu0_local);
  free(hv0_local);

  if (prank == 0){

    // Copying last computed values to h1_, hu1_, hv1_ (if needed)
    if (h0_global != &h1_)
    {
      h1_ = *h0_global;   
      hu1_ = *hu0_global;
      hv1_ = *hv0_global;
    }

    if (output_n > 0)
    {
      writer->add_h(h1_, T);
    }

    std::cout << "Finished solving SWE." << std::endl;
  }
}


void SWESolver::distribute_initial_data(
    double* h0_local, double* hu0_local, double* hv0_local,
    const std::vector<double>& h0, const std::vector<double>& hu0, const std::vector<double>& hv0,
    int nx, int base_rows, int rem,
    int prank, int psize, int p_count) const
{
  if (prank == 0) {
    std::vector<MPI_Request> requests;

    for (int target = 0; target < psize; ++target) {
      int m_local_rows = base_rows + (target < rem ? 1 : 0);
      int m_start_row = base_rows * target + std::min(target, rem);

      int m_extra_top = (target == 0) ? 0 : 1;
      int m_extra_bottom = (target == psize - 1) ? 0 : 1;
      int m_rows_with_ghosts = m_local_rows + m_extra_top + m_extra_bottom;

      int m_copy_start = m_start_row - m_extra_top;
      int m_offset = m_copy_start * nx;
      int m_count = m_rows_with_ghosts * nx;

      const double* h0_buf  = h0.data()  + m_offset;
      const double* hu0_buf = hu0.data() + m_offset;
      const double* hv0_buf = hv0.data() + m_offset;

      if (target == 0) {
        std::memcpy(h0_local,  h0_buf,  m_count * sizeof(double));
        std::memcpy(hu0_local, hu0_buf, m_count * sizeof(double));
        std::memcpy(hv0_local, hv0_buf, m_count * sizeof(double));
      } else {
        MPI_Request req[3];
        MPI_Isend(h0_buf,  m_count, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(hu0_buf, m_count, MPI_DOUBLE, target, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(hv0_buf, m_count, MPI_DOUBLE, target, 2, MPI_COMM_WORLD, &req[2]);
        requests.insert(requests.end(), req, req + 3);
      }
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  } else {
    MPI_Request reqs[3];
    MPI_Irecv(h0_local,  p_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(hu0_local, p_count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(hv0_local, p_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &reqs[2]);
    MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
  }
}

void SWESolver::gather_fields_to_root(const double* h_local,
                                      const double* hu_local,
                                      const double* hv_local,
                                      double* h_global,
                                      double* hu_global,
                                      double* hv_global,
                                      const int nx,
                                      const int psize,
                                      const int base_rows,
                                      const int rem,
                                      const int p_local_rows,
                                      const int p_extra_top) const
{
  std::vector<int> recvcounts(psize);
  std::vector<int> displs(psize);

  // Compute recvcounts and displacements
  for (int i = 0; i < psize; ++i) {
    int local_rows = base_rows + (i < rem ? 1 : 0);
    recvcounts[i] = local_rows * nx;
    displs[i] = (base_rows * i + std::min(i, rem)) * nx;
  }

  const int local_start_idx = p_extra_top * nx;
  const int real_data_count = p_local_rows * nx;

  MPI_Gatherv(h_local  + local_start_idx, real_data_count, MPI_DOUBLE,
              h_global, recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  MPI_Gatherv(hu_local + local_start_idx, real_data_count, MPI_DOUBLE,
              hu_global, recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  MPI_Gatherv(hv_local + local_start_idx, real_data_count, MPI_DOUBLE,
              hv_global, recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
}


void SWESolver::exchange_ghost_rows(double* local,
                         int nx_,
                         int p_local_rows,
                         int prank,
                         int psize,
                         MPI_Comm comm) const 
{
    MPI_Request reqs[4];
    int tag_up = 0, tag_down = 1;

    // Top send (real row 1) -> upper neighbor's ghost (bottom)
    if (prank > 0) {
        MPI_Isend(local + 1 * nx_, nx_, MPI_DOUBLE, prank - 1, tag_up, comm, &reqs[0]);
        MPI_Irecv(local + 0 * nx_, nx_, MPI_DOUBLE, prank - 1, tag_down, comm, &reqs[1]);
    }

    // Bottom send (real row p_local_rows) -> lower neighbor's ghost (top)
    if (prank < psize - 1) {
        MPI_Isend(local + p_local_rows * nx_, nx_, MPI_DOUBLE, prank + 1, tag_down, comm, &reqs[2]);
        MPI_Irecv(local + (p_local_rows + 1) * nx_, nx_, MPI_DOUBLE, prank + 1, tag_up, comm, &reqs[3]);
    }

    // Wait only on the requests that were issued
    int nreqs = 0;
    if (prank > 0) nreqs += 2;
    if (prank < psize - 1) nreqs += 2;

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
}

double
SWESolver::compute_time_step(const double* h,
                             const double* hu,
                             const double* hv,
                             const double T,
                             const double Tend,
                             const int p_local_rows,
                             const int prank) const
{
  double max_nu_sqr = 0.0;
  double au{0.0};
  double av{0.0};

  std::size_t start_col = prank == 0 ? 1 : 2;

  for (std::size_t j = start_col; j < static_cast<std::size_t>(p_local_rows - 1); ++j)
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
                          const double* h0,
                          const double* hu0,
                          const double* hv0,
                          double* h,
                          double* hu,
                          double* hv) const
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

  // h(2:nx-1,2:nx-1) = 0.25*(h0(2:nx-1,1:nx-2)+h0(2:nx-1,3:nx)+h0(1:nx-2,2:nx-1)+h0(3:nx,2:nx-1)) ...
  //     + C1*( hu0(2:nx-1,1:nx-2) - hu0(2:nx-1,3:nx) + hv0(1:nx-2,2:nx-1) - hvhv0:nx,2:nx-1) );

  // hu(2:nx-1,2:nx-1) = 0.25*(hu0(2:nx-1,1:nx-2)+hu0(2:nx-1,3:nx)+hu0(1:nx-2,2:nx-1)+hu0(3:nx,2:nx-1)) -
  // C2*H(2:nx-1,2:nx-1).*Zdx(2:nx-1,2:nx-1) ...
  //     + C1*( hu0(2:nx-1,1:nx-2).^2./h0(2:nx-1,1:nx-2) + 0.5*g*h0(2:nx-1,1:nx-2).^2 -
  //     hu0(2:nx-1,3:nx).^2./h0(2:nx-1,3:nx) - 0.5*g*h0(2:nx-1,3:nx).^2 ) ...
  //     + C1*( hu0(1:nx-2,2:nx-1).*hv0(1:nx-2,2:nx-1)./h0(1:nx-2,2:nx-1) -
  //     hu0(3:nx,2:nx-1).*hv0(3:nx,2:nx-1)./h0(3:nx,2:nx-1) );

  // hv(2:nx-1,2:nx-1) = 0.25*(hv0(2:nx-1,1:nx-2)+hv0(2:nx-1,3:nx)+hv0(1:nx-2,2:nx-1)+hv0(3:nx,2:nx-1)) -
  // C2*H(2:nx-1,2:nx-1).*Zdy(2:nx-1,2:nx-1)  ...
  //     + C1*( hu0(2:nx-1,1:nx-2).*hv0(2:nx-1,1:nx-2)./h0(2:nx-1,1:nx-2) -
  //     hu0(2:nx-1,3:nx).*hv0(2:nx-1,3:nx)./h0(2:nx-1,3:nx) ) ...
  //     + C1*( hv0(1:nx-2,2:nx-1).^2./h0(1:nx-2,2:nx-1) + 0.5*g*h0(1:nx-2,2:nx-1).^2 -
  //     hv0(3:nx,2:nx-1).^2./h0(3:nx,2:nx-1) - 0.5*g*h0(3:nx,2:nx-1).^2  );
}

void
SWESolver::solve_step(const double dt,
                      const double* h0,
                      const double* hu0,
                      const double* hv0,
                      double* h,
                      double* hu,
                      double*hv,
                      const int p_local_rows, 
                      const int prank) const
{
  std::size_t start_row = prank == 0 ? 1 : 2;

  for (std::size_t j = start_row; j < static_cast<std::size_t>(p_local_rows - 1); ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      this->compute_kernel(i, j, dt, h0, hu0, hv0, h, hu, hv);
    }
  }
}

void
SWESolver::update_bcs(const double* h0,
                      const double* hu0,
                      const double* hv0,
                      double* h,
                      double* hu,
                      double* hv,
                      const int p_local_rows,
                      const int prank,
                      const int psize) const
{

  
  const double coef = this->reflective_ ? -1.0 : 1.0;

  // Top Boundary
  if (prank == 0) {
    for (std::size_t i = 0; i < nx_; ++i){
      at(h, i, 0) = at(h0, i, 1);
      at(hu, i, 0) = at(hu0, i, 1);
      at(hv, i, 0) = coef * at(hv0, i, 1);
    }
  }

  //Bottom Boundary
  if (prank == psize - 1) {
    for (std::size_t i = 0; i < nx_; ++i){
      at(h, i, ny_ - 1) = at(h0, i, ny_ - 2);
      at(hu, i, ny_ - 1) = at(hu0, i, ny_ - 2);
      at(hv, i, p_local_rows - 1) = coef * at(hv0, i, p_local_rows - 2);
    }
  }

  std::size_t start_row = prank == 0 ? 0 : 1;

  // Left and right boundaries.
  for (std::size_t j = start_row; j < static_cast<std::size_t>(p_local_rows); ++j)
  {
    at(h, 0, j) = at(h0, 1, j);
    at(h, nx_ - 1, j) = at(h0, nx_ - 2, j);

    at(hu, 0, j) = coef * at(hu0, 1, j);
    at(hu, nx_ - 1, j) = coef * at(hu0, nx_ - 2, j);

    at(hv, 0, j) = at(hv0, 1, j);
    at(hv, nx_ - 1, j) = at(hv0, nx_ - 2, j);
  }
};
