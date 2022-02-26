#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

int main ()
{

  ifstream inFile;

  std::vector<float> x,y;
  std::vector<int> pos;
  double tx,ty;
  int tpos;
  std::string line;

  inFile.open("sm_test0.dat");

  while (std::getline(inFile, line)) {
    inFile >> tx >> ty >> tpos;
    x.push_back(tx);
    y.push_back(ty);
    pos.push_back(tpos);
  }

  inFile.close();
  for(std::vector<int>::size_type i = 0; i != x.size(); i++) {
    cout << x[i] << " " << y[i] << endl;
  }

  return 0;
}
