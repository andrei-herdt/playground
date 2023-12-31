// TODO: Switch to doctest and integrate into production code
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

struct QPProblem {

  QPProblem(size_t nvar, size_t neqconstr) {
    H.resize(nvar, nvar);
    g.resize(nvar);
    A.resize(neqconstr, nvar);
    b.resize(neqconstr);
  }

  Eigen::MatrixXd H;
  Eigen::VectorXd g;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
};


// see ../docs/Carpentier2022.pdf Eq. 4
auto solveLagrangianMM(const QPProblem& qp) -> Eigen::VectorXd {
  constexpr double mue = 0.1;
  Eigen::Vector2d x = Eigen::Vector2d::Zero();
  Eigen::Vector2d y = Eigen::Vector2d::Zero();

  for (size_t i = 0; i < 10; i++) {
    Eigen::MatrixXd Ha = qp.H + 1.0 / mue * qp.A.transpose() * qp.A;
    Eigen::VectorXd ga = qp.g + qp.A.transpose() * (y - mue * qp.b);
    x = -Ha.inverse() * ga;
    y = y + 1.0 / mue * (qp.A * x - qp.b);
  }

  return x;
}

auto computeGradientPDAL(const Eigen::MatrixXd &H, //
                         const Eigen::VectorXd &g, //
                         const Eigen::MatrixXd &A, //
                         const Eigen::VectorXd &b  //
                         ) -> Eigen::VectorXd {
  // Set up KKT matrix
  Eigen::MatrixXd KKT =
      Eigen::MatrixXd::Zero(H.rows() + A.rows(), H.cols() + A.rows());
  Eigen::VectorXd v = Eigen::VectorXd::Zero(H.rows() + A.cols());

  Eigen::MatrixXd I1 = Eigen::MatrixXd::Identity(H.rows(), H.cols());
  Eigen::MatrixXd I2 = Eigen::MatrixXd::Identity(A.rows(), A.rows());

  constexpr double rho = 0.1;
  constexpr double mu_e = 0.1;

  // UL
  KKT.block(0, 0, H.rows(), H.cols()) = H + rho * I1;
  // LL
  KKT.block(H.rows(), 0, A.rows(), A.cols()) = A;
  // UR
  KKT.block(0, H.cols(), A.cols(), A.rows()) = A.transpose();
  // LR
  KKT.block(H.rows(), H.cols(), A.rows(), A.rows()) = -mu_e * I2;
  // TODO: Build KKT matrix and factorize. It is not yet clear how the matrix
  // is being built from the model.
  // TODO: Get solve function from ldl

  Eigen::VectorXd xk = Eigen::VectorXd::Zero(H.rows());
  Eigen::VectorXd yk = Eigen::VectorXd::Zero(A.rows());

  v.head(H.rows()) = rho * xk - g;
  v.tail(A.rows()) = b - mu_e * yk;
  std::cout << "KKT:" << std::endl;
  std::cout << KKT << std::endl;
  std::cout << "v:" << std::endl;
  std::cout << v << std::endl;

  return xk;
}

TEST_CASE("Simple Least Squares", "[ProxQP]") {

  QPProblem qp(2, 2);
  qp.H << 1, 0, 0, 1;
  qp.g << 0, 1;
  qp.A << 1, 0, 0, 1;
  qp.b << 0, -1;

  Eigen::VectorXd result = solveLagrangianMM(qp);

  Eigen::Vector2d expected_result;
  expected_result << 0, -1;
  CHECK(result.isApprox(expected_result, 1e-9));
}
