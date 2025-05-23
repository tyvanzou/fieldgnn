#include "LennardJones.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "GridayException.hpp"

unsigned long long factorial(int n) {
  unsigned long long result = 1;
  for (int i = 1; i <= n; i++) {
    result *= i;
  }
  return result;
}

// 计算组合数 C(n, m)
unsigned long long composite(int n, int m) {
  if (m < 0 || m > n) {
    return 0; // C(n, m) 为零，因为 m 无效
  } else if (m == 0) {
    return 1;
  }

  if (m > n - m) {
    m = n - m; // 优化，减小计算量
  }

  unsigned long long numerator = 1;
  for (int i = n; i > n - m; i--) {
    numerator *= i;
  }

  unsigned long long denominator = factorial(m);

  return numerator / denominator;
}

LennardJones::LennardJones(GReal eps, GReal sig, GReal rcut)
    : PairEnergy{}, mEps{eps}, mSig{sig}, mSigSq{sig * sig}, mRcut{rcut},
      mRcutSq{rcut * rcut} {}

void LennardJones::setSimulationBox(const Cell &box) {
  mBox = box;
  mInvBox = inverse(box);
}

std::vector<GReal> LennardJones::calculate(const Vectors &r1,
                                           const Vectors &r2) {
  GReal zero = static_cast<GReal>(0.001);

  std::vector<GReal> ret(20, 0.0);

  bool cond1 = mEps < zero;
  bool cond2 = mSig < zero;
  bool cond3 = mRcut < zero;

  if (cond1 || cond2 || cond3)
    return ret;

  if (r1.size() == 0 || r2.size() == 0)
    return ret;

  int size1 = r1.size();
  int size2 = r2.size();

  GReal overlapDistSq = 1e-8;
  GReal one = 1.0;

  // for smoothing
  // GReal epsCut = 20.0;
  // std::cout << "eps " << mEps << " sigma " << mSig << " rcut " << mRcut <<
  // std::endl; auto rIn    = pow((2 * (sqrt(1 + epsCut) - 1) / epsCut),
  // (1.0/(GReal)6.0)) * mSig; auto rInSq  = rIn * rIn;

  // std::cout << rIn << std::endl;

  // GReal rInf   = 0.075;
  // auto  rInfSq = rInf * rInf;

  for (int i = 0; i < size1; ++i) {
    const Vector &ri = r1[i];

    for (int j = 0; j < size2; ++j) {
      std::vector<GReal> energy(20, 0.0);

      const Vector &rj = r2[j];

      auto s = mInvBox * (ri - rj);

      for (auto &si : s) {
        if (si > 0.5)
          si -= 1.0;
        if (si < -0.5)
          si += 1.0;
      }

      auto r = mBox * s;
      auto rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

      if (rsq < mRcutSq) {
        if (rsq < overlapDistSq)
          return std::vector<GReal>(20, std::numeric_limits<GReal>::max());

        auto r2 = mSigSq / rsq;
        auto r6 = r2 * r2 * r2;
        auto ri = std::sqrt(rsq);
        auto sig = std::sqrt(mSigSq);

        // pauli
        for (auto i = 0; i <= 11; ++i) {
          auto item = composite(12, i) * std::pow(sig, i) / std::pow(ri, 12);
          energy[i] = item;
        }
        energy[12] = r6 * (r6 - one);
        // london
        for (auto i = 0; i <= 5; ++i) {
          auto item = composite(6, i) * std::pow(sig, i) / std::pow(ri, 6);
          energy[13 + i] = item;
        }
        energy[19] = r6 * (r6 + one);

        for (auto i = 0; i < 20; ++i)
          ret[i] += energy[i];
      }
    }
  }

  // 4.0 * mEps
  for (int i = 0; i < 20; i++) {
    ret[i] *= 4.0 * mEps;
  }

  return ret;
}

std::unique_ptr<PairEnergy> LennardJones::clone() {
  return std::make_unique<LennardJones>(mEps, mSig, mRcut);
}

void LennardJones::print() {
  using namespace std;

  cout << "Pair Type: LennardJones, Parameter: " << setw(10)
       << "Eps = " << setw(10) << mEps << setw(10) << "Sig = " << setw(10)
       << mSig << setw(10) << "Rcut = " << setw(10) << mRcut << endl;
}

std::string LennardJones::getName() { return "LennardJones"; }

GReal LennardJones::getEps() { return mEps; }

GReal LennardJones::getSig() { return mSig; }

GReal LennardJones::getRcut() { return mRcut; }

template <>
PairEnergy::PairEnergyPtr mixPairEnergy<LennardJones>(PairEnergy &e1,
                                                      PairEnergy &e2) {
  LennardJones *p1 = nullptr;
  LennardJones *p2 = nullptr;

  p1 = dynamic_cast<LennardJones *>(&e1);
  p2 = dynamic_cast<LennardJones *>(&e2);

  if (p1 == nullptr or p2 == nullptr)
    THROW_EXCEPT("Invalid mixing occurs");

  double eps = std::sqrt(p1->getEps() * p2->getEps());
  double sig = 0.5 * (p1->getSig() + p2->getSig());
  double rcut = 0.5 * (p1->getRcut() + p2->getRcut());

  return std::make_unique<LennardJones>(eps, sig, rcut);
}
