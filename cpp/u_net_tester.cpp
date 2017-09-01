#include <bits/stdc++.h>
#define ADD(a, b) a = (a + ll(b)) % mod
#define MUL(a, b) a = (a * ll(b)) % mod
#define MAX(a, b) a = max(a, b)
#define MIN(a, b) a = min(a, b)
#define rep(i, a, b) for(int i = (a); i < (b); i++)
#define rer(i, a, b) for(int i = (a) - 1; i >= (b); i--)
#define all(a) (a).begin(), (a).end()
#define sz(v) (int)(v).size()
#define pb push_back
#define sec second
#define fst first
#define debug(fmt, ...) Debug(__LINE__, ":", fmt, ##__VA_ARGS__)
using namespace std;
typedef long long ll;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
typedef pair<int, pi> ppi;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<vl> mat;
typedef complex<double> comp;
void Debug() {cout << '\n'; }
template<class FIRST, class... REST>void Debug(FIRST arg, REST... rest) { 
	cout << arg << " "; Debug(rest...); }
template<class T>ostream& operator<< (ostream& out, const vector<T>& v) {
	out << "[";if(!v.empty()){rep(i,0,sz(v)-1)out<<v[i]<< ", ";out<<v.back();}out << "]";return out;}
template<class S, class T>ostream& operator<< (ostream& out, const pair<S, T>& v) {
	out << "(" << v.first << ", " << v.second << ")";return out;}
const int MAX_N = 200010;
const int MAX_V = 100010;
const double eps = 1e-6;
const ll mod = 1000000007;
const int inf = 1 << 29;
const ll linf = 1LL << 60;
const double PI = 3.14159265358979323846;
///////////////////////////////////////////////////////////////////////////////////////////////////

vector<pi> vec;

pi unet(int n) {
	bool ok = true;
	rep(i, 0, 4) {
		n -= 4;
		if(n < 0 || n % 2 != 0) {
			ok = false; 
			break;
		}
		n /= 2;
	}
	if(!ok || n <= 10) return pi(-1, -1);
	else {
		int a = n;
		n -= 4;
		rep(i, 0, 4) {
			n *= 2;
			n -= 4;
		}
		return pi(a, n);
	}
}

void solve() {
	vector<ppi> vec;
	for(int i = 1; i <= 1000; i++) {
		pi a = unet(i);
		if(a != pi(-1, -1)) vec.pb(ppi(i, a));
	}
	debug(vec);
}

int main() {
	solve();
    cerr << "Time elapsed: " << 1.0 * clock() / CLOCKS_PER_SEC << " s.\n";
	return 0;
}


