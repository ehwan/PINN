#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

// ========================= Global Settings =========================
using scalar_type = double;

template <int N>
using vecN = Eigen::Vector<scalar_type, N>;

using ivec = Eigen::Vector<int, 2>;
using vec = Eigen::Vector<scalar_type, 2>;

constexpr int Width = 100;
constexpr int Height = 100;
const scalar_type Re = 100.0;

ivec idx2dir(int idx)
{
  assert(idx < 9);
  int x = idx % 3;
  int y = idx / 3;

  return { x - 1, y - 1 };
}
int dir2idx(ivec dir)
{
  return dir.x() + 1 + (dir.y() + 1) * 3;
}
scalar_type weight(ivec dir)
{
  int manhat = std::abs(dir.x()) + std::abs(dir.y());
  assert(manhat < 3);
  if (manhat == 0)
  {
    return 4.0 / 9.0;
  }
  if (manhat == 1)
  {
    return 1.0 / 9.0;
  }
  if (manhat == 2)
  {
    return 1.0 / 36.0;
  }
  return 0;
}
scalar_type weight(int idx)
{
  return weight(idx2dir(idx));
}

struct LidDrivenCavityLBM
{
  int width;
  scalar_type Re;
  scalar_type dx, dt;

  // nondimensionalized
  scalar_type tau;

  // nondimensionalized
  scalar_type u0;

  std::vector<vecN<9>> f, ftemp;
  std::vector<vec> velocity;
  std::vector<Eigen::Vector<float, 2>> vel_dim;
  std::vector<scalar_type> density;

  template <typename T>
  T& at(std::vector<T>& v, int x, int y)
  {
    return v[x + y * width];
  }
  template <typename T>
  T& at(std::vector<T>& v, ivec p)
  {
    return at(v, p.x(), p.y());
  }

  bool valid(int x, int y) const
  {
    return x >= 0 && x < width && y >= 0 && y < width;
  }
  bool valid(ivec v) const
  {
    return valid(v.x(), v.y());
  }

  void init(int w, scalar_type re, scalar_type dt)
  {
    width = w;
    dx = 1.0 / (width - 1);
    this->dt = dt;
    Re = re;

    const int wh = width * width;
    f.resize(wh);
    ftemp.resize(wh);
    velocity.resize(wh);
    density.resize(wh);
    vel_dim.resize(wh);

    scalar_type nu = 1.0 / Re;
    // nondim
    nu /= (dx * dx / dt);
    u0 = 1.0 / (dx / dt);

    tau = (6 * nu + 1) / 2.0;

    std::cout << "W: " << width << "\n";
    std::cout << "Re: " << re << "\n";
    std::cout << "U0: " << u0 << "\n";
    std::cout << "tau: " << tau << "\n";
    std::cout << "dt: " << dt << "\n";

    for (int y = 0; y < width; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        if (y == width - 1)
        {
          at(vel_dim, x, y) = vec(1, 0).cast<float>();
          at(velocity, x, y) = vec(u0, 0);
        }
        else
        {
          at(vel_dim, x, y) = vec(0, 0).cast<float>();
          at(velocity, x, y) = vec(0, 0);
        }
        at(density, x, y) = 1;
        at(f, x, y) = equilibrium(at(velocity, x, y), at(density, x, y));
      }
    }
  }

  void step()
  {
#define INVOKE(name)                \
  for (int y = 0; y < width; ++y)   \
  {                                 \
    for (int x = 0; x < width; ++x) \
    {                               \
      name(x, y);                   \
    }                               \
  }
    INVOKE(streaming);
    boundary_condition();
    INVOKE(macroscopic);
    INVOKE(equilibrium);
    INVOKE(collision);
    for (int y = 0; y < width; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        at(vel_dim, x, y) = get_velocity(x, y).cast<float>();
      }
    }
    // streaming  f -> ftemp
    // boundary condition ( each side )
    // macroscopic ftemp -> density, velocity
    // equilibrium density, velocity -> f
    // collision f(feq), ftemp(fstar) -> f
  }
  vecN<9> equilibrium(vec vel, scalar_type dens)
  {
    vecN<9> ret;
    for (int i = 0; i < 9; ++i)
    {
      vec dir = idx2dir(i).cast<scalar_type>();
      scalar_type edotu = dir.dot(vel);

      scalar_type feq = weight(i) * dens
                        * (1.0 + 3.0 * edotu + 4.5 * edotu * edotu
                           - 1.5 * vel.squaredNorm());
      ret[i] = feq;
    }
    return ret;
  }
  void streaming(int x, int y)
  {
    for (int dir = 0; dir < 9; ++dir)
    {
      ivec dirvec = idx2dir(dir);
      ivec negdir = -dirvec;
      ivec adjpos = { x, y };
      adjpos += dirvec;

      if (valid(adjpos) == false)
      {
        continue;
      }

      at(ftemp, x, y)[dir2idx(negdir)] = at(f, adjpos)[dir2idx(negdir)];
    }
  }
  void boundary_condition()
  {
    // upper moving wall
    for (int x = 0; x < width; ++x)
    {
      vec vel = { u0, 0 };
      scalar_type dens = at(density, x, width - 2);
      at(velocity, x, width - 1) = vel;
      at(density, x, width - 1) = dens;
      at(ftemp, x, width - 1) = equilibrium(vel, dens);
    }

    // lower wall
    for (int x = 0; x < width; ++x)
    {
      for (int dir = 0; dir < 9; ++dir)
      {
        ivec dirvec = idx2dir(dir);
        ivec negdir = -dirvec;
        ivec adjpos = { x, 0 };
        adjpos += dirvec;
        if (valid(adjpos) == false)
        {
          at(ftemp, x, 0)[dir2idx(negdir)] = at(ftemp, x, 0)[dir];
        }
      }
    }

    // left right
    for (int y = 1; y < width - 1; ++y)
    {
      for (int x : { 0, width - 1 })
      {
        for (int dir = 0; dir < 9; ++dir)
        {
          ivec dirvec = idx2dir(dir);
          ivec negdir = -dirvec;
          ivec adjpos = { x, y };
          adjpos += dirvec;
          if (valid(adjpos) == false)
          {
            at(ftemp, x, y)[dir2idx(negdir)] = at(ftemp, x, y)[dir];
          }
        }
      }
    }
  }
  void macroscopic(int x, int y)
  {
    scalar_type dens = 0;
    vec vel = { 0, 0 };
    for (int i = 0; i < 9; ++i)
    {
      vec dir = idx2dir(i).cast<scalar_type>();

      dens += at(ftemp, x, y)[i];
      vel += dir * at(ftemp, x, y)[i];
    }
    vel /= dens;
    at(density, x, y) = dens;
    at(velocity, x, y) = vel;
  }
  void equilibrium(int x, int y)
  {
    const vec vel = at(velocity, x, y);
    const scalar_type dens = at(density, x, y);
    at(f, x, y) = equilibrium(vel, dens);
  }

  void collision(int x, int y)
  {
    // collision f(feq), ftemp(fstar) -> f
    auto fstar = at(ftemp, x, y);
    auto feq = at(f, x, y);

    fstar = fstar - (fstar - feq) / tau;
    at(f, x, y) = fstar;
  }
  vec get_velocity(int x, int y)
  {
    return at(velocity, x, y) / u0;
  }
};

int main(int argc, char** argv)
{
  LidDrivenCavityLBM cavity;
  // u0 = 1/ (dx / dt) < 0.25
  // dt < 0.25 * dx
  // W = 200 can cover Re \in (20, 1000) // kolmogorov scale

  // dt = 0.25 * 0.005 = 1/200 / 4 = 1/800

  cavity.init(200, 2000.0, 0.25 * 0.005);

  for (int i = 0; i < 1000; ++i)
  {
    cavity.step();
  }

  std::ofstream out("out.dat", std::ios::binary);
  out.write((char*)cavity.vel_dim.data(),
            sizeof(float) * cavity.width * cavity.width * 2);
  return 0;
}