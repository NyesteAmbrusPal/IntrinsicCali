#include "functional_impl.h"

int gcd(int a, int b) {
	int R;
	while ((a % b) > 0) {
		R = a % b;
		a = b;
		b = R;
	}
	return b;
}

int find_divisor(int& a, int& b, int limit, int max)
{
	int divisor = 2;
	int number = a - b;
	while (((number % divisor) != 0) || (divisor <= limit))
	{
		divisor++;
		if (divisor > max)
		{
			a = a - 1;
			b = b + 1;
			number = a - b;
			divisor = 2;
		}
	}
	return divisor;
}

template <typename ITER, typename T>
ITER look_for(ITER first, ITER last, const T& value)
{
	for (ITER it = first; it != last; ++it)
	{
		if (*it == value)
		{
			return it;
		}
	}
	return last;
}