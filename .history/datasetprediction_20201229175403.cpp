#include<bits/stdc++.h>
#define int long long
using namespace std;
signed main()
{
    double a[]={-569.6,-569.6,-1780.0,-1780.0};
    double b[]={-530.9,530.9,-530.9,530.9};
    double c[]={-890.0,-1411.0,-1560.0,-2220.0,-2091.0,-2878.0,-3537.0,-3268.0,-3920.0,-4163.0,-5471.0,-5157.0};
    double Min=1e8;
    int ind;
    for(int i=0;i<4;i++)
    {
        double f=a[i];
        double s=b[i];
        double cal=0;
        for(int j=1;j<=10;j++)
        {
            double res=s*j+f;
            res=res-c[j-1];
            res=res*res;
            cal+=res;
        }
        cout<<cal<<endl;
        if(cal<Min)
        {
            ind=i;
            Min=cal; 
        }
    }
    cout<<ind<<" "<<Min<<endl;
    return 0;
}
