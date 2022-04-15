#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <functional>
using namespace std;

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void display(vector<float> nums)

{

        //Print the values of the vector using loop

        for(auto ele = nums.begin(); ele != nums.end() ; ele++)

        cout << *ele << " ";

        //Add new line

        cout << "\n";

}

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      int start1  = (x - hx <= 0) ? 0 : x - hx;
      int end1 = (x + hx + 1 >= nx) ? nx : x + hx +1;
      int start2 = (y - hy <= 0) ? 0 : y - hy ;
      int end2 = (y + hy + 1 >= ny) ? ny : y + hy +1;

      std::array<float> win((end1 - start1) * (end2 - start2));

      int p = 0;
      for (int i = start2; i < end2; i++) {
				for (int j = start1; j < end1; j++) {
					win[p++] = in[j + i * nx];
				}
			}

      float median = 0;
      int s = 0, s2 = 0;
      s = win.size();
      s2 = s / 2;
      auto m = win.begin() + s2;
			std::nth_element(win.begin(), m, win.end());
			median = win[s2];
    
      if (s % 2 == 0) {
				std::nth_element(win.begin(), m - 1, win.end());
				median = (median + win[s2 - 1]) / 2.0;
			} 
			out[x + y * nx] = median;
		
    }
  }
}
