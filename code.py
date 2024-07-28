from math import pi, exp


# Tolerance under 5 sigma
tol = 1e-8

# Constants
H = 2.18 * 1e-18
G = 6.67430 * 1e-11
c = 299_792_458
h = 6.62607015 * 1e-34
hbar = h / (2 * pi)

# K parameter as a function of the mass m
def K(m):
  return (5120 * pi * (G ** 2) * (m ** 3)) / (hbar * (c ** 4))

# Schwarzschild radius
def rs(m):
  return 2 * G * m / (c ** 2)


# Newton method
def newton(K, H, initial_guess=1.0, tol=tol, max_iter=100_000):
  t = initial_guess
  for _ in range(max_iter):
    f_t = t - K * exp(3 * H * t)
    f_prime_t = 1 - 3 * H * K * exp(3 * H * t)
    t_next = t - f_t / f_prime_t
    
    if abs(t_next - t) < tol:
      return t_next
    
    t = t_next
  raise ValueError('Newton method did not converge')

# Define the bisection method
def newton_hawking_bisection(H, initial_guess, tol=tol, max_iter=100_000, m_low=1_000, m_high=1_000_000):
  def condition(m):
    k = K(m)
    t_solution = newton(k, H, initial_guess)
    return k < t_solution

  low = m_low
  high = m_high
  mid = (low + high) / 2

  for _ in range(max_iter):
    if condition(mid):
      high = mid
    else:
      low = mid
    mid = (low + high) / 2

    if abs(high - low) < tol:
      return mid

  raise ValueError('Bisection method did not converge')


# Initial t to start from
t_initial_guess = 1

# Looking for the cricial mass M_0
m_critical = newton_hawking_bisection(H, t_initial_guess)

# Output results
print(f'The critical mass value where k < t_solution switches from False to True is approximately: {m_critical:.5e} kg')
print(f'Value of t for a = 1 given the obtained mass: {K(m_critical):.5f} s')
print(f'Schwarzschild radius for this result: {rs(m_critical):.5e} m')
