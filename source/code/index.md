---
title: 模板库
date: 2024-02-02 19:27:22
type: "code"
layout: "code"
---

#### 高精度

##### 高精加

```cpp
#include<bits/stdc++.h>
using namespace std;
vector<int> add(vector<int> A,vector<int> B){
    vector<int> C;
    int t = 0;
    for(int i = 0;i < A.size() || i < B.size();i ++){
        if(i < A.size())t += A[i];
        if(i < B.size())t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    if(t)C.push_back(1);
    return C;
}
int main(){
    string a,b;
    cin>>a>>b;
    vector<int> A,B;
    for(int i = a.size() - 1;i >= 0;i --)A.push_back(a[i] - '0');
    for(int i = b.size() - 1;i >= 0;i --)B.push_back(b[i] - '0');
    vector<int> C = add(A,B);
    for(int i = C.size() - 1;i >= 0;i --)cout<<C[i];
    return 0;
}
```

##### 高精乘低精

```cpp
#include<bits/stdc++.h>
using namespace std;
vector<int> mul(vector<int> A,int b){
    vector<int> C;
    int t = 0;
    if(b == 0){
        C.push_back(0);
        return C;
    }
    for(int i = 0;i < A.size() || t;i ++){
        if(i < A.size())t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    return C;
}
int main(){
    string a;
    int b;
    cin>>a>>b;
    vector<int> A;
    for(int i = a.size() - 1;i >= 0;i --)A.push_back(a[i] - '0');
	vector<int> C = mul(A,b);
    for(int i = C.size() - 1;i >= 0;i --)cout<<C[i];
    return 0;
}
```

##### 高精乘高精

```cpp
#include<bits/stdc++.h>
using namespace std;
vector<int> mul(vector<int> A,int b){
  	vector<int> C;
  	int t = 0;
    if(b == 0){
        C.push_back(0);
        return C;
    }
  	for(int i = 0; i < A.size() || t;i++){
      	if(i < A.size())t += A[i] * b;
      	C.push_back(t % 10);
      	t /= 10;
    }
  	return C;
}
vector<int> add(vector<int> A,vector<int> B)
{
  	vector<int> C;
  	int t = 0;
  	for(int i = 0; i < A.size() || i < B.size(); i++){
        if(i < A.size())t += A[i];
        if(i < B.size())t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    if(t) C.push_back(1);
    return C;
}
int main(){
    string a,b;
    cin>>a>>b;
    vector<int> A;
    for(int i = a.size() - 1;i >= 0;i --)A.push_back(a[i] - '0');
    vector<int> sum,C;
    sum.push_back(0);
    for(int i = 0;i < b.size();i ++){
        int k = b[i] - '0';
        C = mul(A,k);
        sum = mul(sum,10);
        sum = add(sum,C);
    }
    for(int i = sum.size() - 1;i >= 0;i --){
        cout<<sum[i];
    }
    return 0;
}
```

##### 高精减

```cpp
#include<bits/stdc++.h>
using namespace std;
bool cmp(vector<int> A, vector<int> B)
{
    if(A.size() != B.size()) return A.size() > B.size();
    for(int i = A.size() - 1; i >= 0; i--)
        if(A[i] != B[i]) return A[i] > B[i];
    return true;
}
vector<int> sub(vector<int> A, vector<int> B )
{
    vector<int> C;
    int t = 0;
    for(int i = 0; i < A.size(); i++)
    {
        t = A[i] - t;
        if(i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if(t < 0) t = 1;
        else t = 0;
    }
    while(C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
int main(){
    string a,b;
    cin>>a>>b;
    vector<int> A,B;
    for(int i = a.size() - 1;i >= 0;i --){
        A.push_back(a[i] - '0');
    }
    for(int i = b.size() - 1;i >= 0;i --){
        B.push_back(b[i] - '0');
    }
    if(cmp(A,B) == 0){
        vector<int> F = A;
        A = B;
        B = F;
        cout<<"-";
	}
    vector<int> C = sub(A,B);
    
    for(int i = C.size() - 1;i >= 0;i --){
        cout<<C[i];
    }
    return 0;
}
```

##### 高精除

```cpp
#include<bits/stdc++.h>
using namespace std;
vector<int> div(vector<int> A, int b, int r){
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- ) //倒序，主函数倒一次，这里再倒着处理，实现从最高位计算
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end()); //得到的商是正向的，reverse将商倒序，方便主函数输出（入乡随俗，跟其它三种保持一致）
    while (C.size() > 1 && C.back() == 0) C.pop_back(); //去掉前导0
    return C;
}
int main(){
    string a;
    int b;
    cin>>a>>b;
    vector<int> A;
    for(int i = a.size() - 1;i >= 0;i --)A.push_back(a[i] - '0');
    vector<int> C = div(A,b,0);
    for(int i = C.size() - 1;i >= 0;i --)cout<<C[i];
    return 0;
}
```

#### 快速幂

P1226

```cpp
#include<bits/stdc++.h>
using namespace std;
long long a,b,p;
long long ksm(long long a,long long b,long long p){
	long long ans = 1;
	while(b){
		if(b & 1)
			ans *= a,ans %= p;
		a *= a,a %= p;
		b >>= 1;
	}
	return ans;
}
int main(){
	cin>>a>>b>>p;
	cout<<a<<"^"<<b<<" mod "<<p<<"="<<ksm(a,b,p);
	return 0;
}
```

#### 朴素并查集

P3367

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 1e4 + 20;
int f[N],n,m;;
void init(){
	for(int i = 1;i <= n;i ++)
		f[i] = i;
}
int find(int x){
	if(f[x] == x)
		return x;
	return f[x] = find(f[x]);
}
void join(int x,int y){
	int fx = find(x),fy = find(y);
	if(fx != fy)f[fx] = fy;
}
int main(){
	cin>>n>>m;
	init();
	while(m --){
		int z,x,y;
		cin>>z>>x>>y;
		if(z == 1)
			join(x,y);
		else {
			if(find(x) == find(y))
				cout<<"Y"<<endl;
			else cout<<"N"<<endl;
		}
	}
	return 0;
}
```

#### 快速排序

P1177

```cpp
#include<bits/stdc++.h>
using namespace std;
void qsort(int a[],int l,int r){
    int i = l,j = r,flag = a[(l + r) / 2],t;
    do{
        while(a[i] < flag)i ++;
        while(a[j] > flag)j --;
        if(i <= j){
            t = a[i],a[i] = a[j],a[j] = t;
            i ++,j --;
        }
    }while(i <= j);
    if(i < r)qsort(a,i,r);
    if(j > l)qsort(a,l,j);
}
int n,a[1000000];
int main(){
    cin>>n;
    for(int i = 0;i < n;i ++)
        cin>>a[i];
    qsort(a,0,n - 1);
    for(int i = 0;i < n;i ++)
        cout<<a[i]<<" ";
    return 0;
}
```

#### 线性筛素数

P3383

```cpp
#include<bits/stdc++.h>
using namespace std;
int n,q,x,pri[100000005] = {0};
void linear_sieve(int n){
    bool p[100000005];
    int cnt = 0;
    p[0] = p[1] = 1;
    for(int i = 2; i <= n; i++){
        if(!p[i]) pri[++cnt] = i; 
        for(int j = 1; j <= cnt && i * pri[j] <= n; j ++){
            p[i * pri[j]] = 1;
            if(i % pri[j] == 0) break;
        }
    }
}
int main(){
    cin>>n>>q;
    linear_sieve(n);
    while(q --){
        scanf("%d",&x);
        printf("%d\n",pri[x]);
    }
    return 0;
}
```

#### 拓扑排序

B3644

```cpp
#include<bits/stdc++.h>
using namespace std;
int n,dep[120];
vector<int> g[120];
void toopsort(){
    queue<int> q;
    for(int i = 1;i <= n;i ++)
        if(dep[i] == 0){
            cout<<i<<" ";
            q.push(i);
        }
    while(q.size()){
        int x = q.front();
        q.pop();
        for(auto it:g[x]){
            dep[it] --;
            if(dep[it] == 0){
                cout<<it<<" ";
                q.push(it);
            }
        }
    }
}
int main(){
    cin>>n;
    for(int i = 1;i <= n;i ++){
        int x;
        while(cin>>x){
            if(x == 0)
                break;
            dep[x] ++;
            g[i].push_back(x);
        }
    }
    toopsort();
    return 0;
}
```

#### ST 表

P3865

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 1e6 + 20;
int m,n,f[N][25];
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    cin>>m>>n;
    for(int i = 1;i <= m;i ++)
        cin>>f[i][0];
    for(int j = 1;j <= 21;j ++)
        for(int i = 1;i + (1 << j) - 1 <= m;i ++)
            f[i][j] = max(f[i][j - 1],f[i + (1 << (j - 1))][j - 1]);
    while(n --){
        int l,r;
        cin>>l>>r;
        int k = __lg(r - l + 1);
        cout<<max(f[l][k],f[r - (1 << k) + 1][k])<<"\n";
    }
    return 0;
}
```

#### 单调栈

P5788

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 3e6 + 20;
int n,a[N],f[N];
stack<int> s;
int main(){
	cin>>n;
	for(int i = 1;i <= n;i ++)
		cin>>a[i];
	for(int i = n;i >= 1;i --){
		while(s.size() && a[s.top()] <= a[i])
			s.pop();
        f[i] = s.size() ? s.top() : 0;
		s.push(i);
	}
    for(int i = 1;i <= n;i ++)
        cout<<f[i]<<" ";
	return 0;
}
```

#### 单调队列

P5788

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 1e6 + 20;
deque<int> dq,qd;
int n,k,v[N];
vector<int> Min,Max;
int main(){
	cin>>n>>k;
	for(int i = 1;i <= n;i ++)
		cin>>v[i];
	for(int i = 1;i <= n;i ++){
		if(dq.size() && i - dq.front() >= k)
			dq.pop_front();
		if(qd.size() && i - qd.front() >= k)
			qd.pop_front();
		while(dq.size() && v[dq.back()] > v[i])
			dq.pop_back();
		while(qd.size() && v[qd.back()] < v[i])
			qd.pop_back();
		dq.push_back(i);
		qd.push_back(i);
		if(i >= k){
			Min.push_back(v[dq.front()]);
			Max.push_back(v[qd.front()]);
		}
	}
	for(auto it:Min)
		cout<<it<<" ";
	cout<<endl;
	for(auto it:Max)
		cout<<it<<" ";
	return 0;
}
```

#### 最小生成树

P3366

##### Kruskal

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 5e3 + 20;
int n,m,fa[N],cnt;
long long ans;
struct edge{
	int u,v,w;
};
vector<edge> g;
bool cmp(edge a,edge b){
	return a.w < b.w;
}
void init(){
	for(int i = 1;i <= n;i ++)
		fa[i] = i;
}
int find(int x){
	if(fa[x] == x)
		return x;
	return fa[x] = find(fa[x]);
}
void kruskal(){
	sort(g.begin(),g.end(),cmp);
	for(auto it:g){
		int fu = find(it.u),fv = find(it.v);
		if(fu == fv)
			continue;
		ans += it.w;
		cnt ++;
		fa[fu] = fv;
		if(cnt == n - 1)
			break;
	} 
}
int main(){
	cin>>n>>m;
	init();
	while(m --){
		int u,v,w;
		cin>>u>>v>>w;
		g.push_back({u,v,w});
	}
	kruskal();
	if(cnt != n - 1)
		cout<<"orz";
	else cout<<ans;
}
```

##### prim
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 5e3 + 20,M = 4e5 + 20;
struct edge{
    int u,v,w,nxt;
}e[M];
struct node{
    int l,u;
    bool operator<(node t)const{
        return l > t.l;
    }
};
int n,m,head[N],cnt,tot,dis[N],ans;
bool vis[N];
void add(int u,int v,int w){
    e[++ cnt] = {u,v,w,head[u]};
    head[u] = cnt;
}
void prim(){
    fill(dis + 1,dis + n + 1,0x3f3f3f3f);
    priority_queue<node> pq;
    pq.push({0,1});
    dis[1] = 0;
    while(pq.size()){
        node t = pq.top();
        pq.pop();
        if(vis[t.u])
            continue;
        vis[t.u] = true;
        ans += t.l;
        tot ++;
        int u = t.u;
        for(int i = head[u];i;i = e[i].nxt){
            int v = e[i].v,w = e[i].w;
            if(dis[v] > w){
                dis[v] = w;
                pq.push({dis[v],v});
            }
        }
    }
}
int main(){
    cin>>n>>m;
    while(m --){
        int u,v,w;
        cin>>u>>v>>w;
        add(u,v,w);
        add(v,u,w);
    }
    prim();
    if(tot < n)
        cout<<"orz";
    else cout<<ans;
    return 0;
}
```

#### LCA

P3379

##### 倍增

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 5e5 + 20;
int n,m,s,f[N][25],dep[N];
vector<int> g[N];
void dfs(int u,int fa){
	dep[u] = dep[fa] + 1;
	f[u][0] = fa;
	for(int i = 1;(1 << i) <= dep[u];i ++)
		f[u][i] = f[f[u][i - 1]][i - 1];
	for(auto it:g[u])
		if(it != fa)
			dfs(it,u);
}
int lca(int x,int y){
	if(dep[x] < dep[y])
		swap(x,y);
	int t = dep[x] - dep[y];
	for(int i = 21;i >= 0;i --)
		if(t & (1 << i))
			x = f[x][i];
	if(x == y)
		return x;
	for(int i = 21;i >= 0;i --)
		if(f[x][i] != f[y][i]){
			x = f[x][i];
			y = f[y][i];
		}
	return f[x][0];
}
int main(){
	cin>>n>>m>>s;
	for(int i = 1;i < n;i ++){
		int x,y;
		cin>>x>>y;
		g[x].push_back(y);
		g[y].push_back(x);
	}
	dfs(s,0);
	while(m --){
		int x,y;
		cin>>x>>y;
		cout<<lca(x,y)<<endl;
	}
	return 0;
}
```

#### 模意义下的乘法逆元

P3811

```cpp
#include<bits/stdc++.h>
using namespace std;
long long n,p,inv[3000005];
int main(){
    cin>>n>>p;
    inv[1] = 1;
    cout<<1<<endl;
    for(int i = 2; i <= n;i ++){
        inv[i] = (p - p / i) * inv[p % i] % p;
        printf("%lld\n",inv[i]);
    }
    return 0;
}
```

#### 单源最短路

P4779 

##### Dijkstra

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 20;
struct edge{
	int v,w;
};
struct node{
	int l,u;
	bool operator<(const node t) const{
		return l > t.l;
	}
};
vector<edge> g[N];
vector<int> ans(N,INT_MAX);
int n,m,s;
void dijkstra(int s){
	priority_queue<node> pq;
	pq.push({0,s});
	ans[s] = 0;
	while(pq.size()){
		node f = pq.top();
		pq.pop();
		if(f.l > ans[f.u])
			continue;
		for(auto it:g[f.u])
			if(f.l + it.w < ans[it.v]){
				ans[it.v] = f.l + it.w;
				pq.push({ans[it.v],it.v});
			}
	}
}
int main(){
	cin>>n>>m>>s;
	while(m --){
		int u,v,w;
		cin>>u>>v>>w;
		g[u].push_back({v,w});
	}
	dijkstra(s);
	for(int i = 1;i <= n;i ++)
		cout<<ans[i]<<" ";
	return 0;
} 
```
##### SPFA
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 20;
struct edge{
	int v,w;
};
int n,m,s,dis[N];
bool note[N];
vector<edge> g[N];
void spfa(int s){
	for(int i = 1;i <= n;i ++)
		dis[i] = INT_MAX;
	dis[s] = 0;
	note[s] = true;
	queue<int> q;
	q.push(s);
	while(q.size()){
		int u = q.front();
		q.pop();
		note[u] = false;
		for(auto it:g[u]){
			int v = it.v;
			int w = it.w;
			if(dis[v] > dis[u] + w){
				dis[v] = dis[u] + w;
				if(!note[v]){
					q.push(v);
					note[v] = true;
				}
			}
		}
	}
}
int main(){
	cin>>n>>m>>s;
	while(m --){
		int u,v,w;
		cin>>u>>v>>w;
		g[u].push_back({v,w});
	}
	spfa(s);
	for(int i = 1;i <= n;i ++)
		cout<<dis[i]<<" ";
	return 0;
}
```

#### 差分
##### 一维差分
$l$ 加 $k$，$r + 1$ 减 $k$。

##### 点差分
P3258

$x$ 与 $y$ 加 $k$，lca 与其父亲减 $k$。

节点的前缀和是其所有子节点的前缀和之和。
##### 边差分
CF191C

$x$ 与 $y$ 加 $k$，lca 减 $k$。

节点的前缀和是其所有子节点的前缀和之和。
#### 链式前向星
##### 定义
```cpp
struct Edge{
	int to,nxt,w; // to 为终点，nxt 表示前一个以 u 为起点的边的编号，w 为边权
}e[M]; // M 为边个数
int head[N],cnt;
```
##### 加边
```cpp
void add(int u,int v,int w){
	e[++ cnt].to = v;
	e[cnt].w = w;
	e[cnt].nxt = head[u];
	head[u] = cnt;
}
```
##### 遍历以 $u$ 为起点的所有边
```cpp
for(int i = head[u];i != 0;i = e[i].nxt)
  // Do some tings...
```

#### 次短路
P2865
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 5e3 + 20;
struct edge{
	int v,w;
};
struct node{
	int l,u;
	const bool operator<(const node t) const{
		return l > t.l;
	}
};
int n,m,dis[N][2];
vector<edge> g[N];
void dij(int s){
	for(int i = 1;i <= n;i ++)
		dis[i][0] = dis[i][1] = 0x3f3f3f3f;
	priority_queue<node> pq;
	pq.push({0,s});
	dis[s][0] = 0;
	while(pq.size()){
		node f = pq.top();
		pq.pop();
		int u = f.u,d = f.l;
		if(d > dis[u][1])
			continue;
		for(auto it:g[u]){
			int v = it.v;
			int w = it.w;
			if(dis[v][0] > d + w){
				dis[v][1] = dis[v][0];
				dis[v][0] = d + w;
				pq.push({dis[v][0],v});
			}
			if(dis[v][0] < d + w && dis[v][1] > d + w){
				dis[v][1] = d + w;
				pq.push({dis[v][1],v});
			}
		}
	}
}
int main(){
	cin>>n>>m;
	while(m --){
		int u,v,w;
		cin>>u>>v>>w;
		g[u].push_back({v,w});
		g[v].push_back({u,w});
	}
	dij(1);
	cout<<dis[n][1];
	return 0;
}
```
#### 差分约束
P5960
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 5e3 + 20;
struct edge{
	int v,w;
};
int n,m,dis[N],d[N];
bool vis[N];
vector<edge> g[N];
void spfa(int s){
	for(int i = 1;i <= n;i ++)
		dis[i] = 0x3f3f3f3f;
	queue<int> q;
	q.push(s);
	dis[s] = 0;
	vis[s] = true;
	d[s] = 1;
	while(q.size()){
		int u = q.front();
		q.pop();
		vis[u] = false;
		for(auto it:g[u]){
			int v = it.v;
			int w = it.w;
			if(dis[v] > dis[u] + w){
				dis[v] = dis[u] + w;
				if(!vis[v]){
					vis[v] = true;
					d[v] ++;
					q.push(v);
					if(d[v] >= n + 1){
						cout<<"NO";
						exit(0);
					}
				}
			}
		}
	} 
}
int main(){
	cin>>n>>m;
	for(int i = 1;i <= n;i ++)
		g[0].push_back({i,0});
	while(m --){
		int u,v,w;
		cin>>u>>v>>w;
		g[v].push_back({u,w});
	}
	spfa(0);
	for(int i = 1;i <= n;i ++)
		cout<<dis[i]<<" ";
	return 0;
}
```
#### tarjan 强连通分量
P2341
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 2e5 + 20,M = 3e5 + 20;
struct edge{
	int u,v,nxt;
}e[M];
int n,m,head[N],cnt,tot,sum,res,ans,belong[N],num[N],dfn[N],low[N],b[N],out[N];
stack<int> s;
void add(int u,int v){
	e[++ cnt] = {u,v,head[u]};
	head[u] = cnt;
}
void tarjan(int u){
	dfn[u] = low[u] = ++tot;
	s.push(u);
	b[u] = true;
	for(int i = head[u];i;i = e[i].nxt){
		int v = e[i].v;
		if(!dfn[v]){
			tarjan(v);
			low[u] = min(low[u],low[v]);
		}
		else if(b[v])
			low[u] = min(low[u],dfn[v]);
	}
	if(dfn[u] == low[u]){
		++ sum;
		while(true){
			int v = s.top();
			s.pop();
			b[v] =false;
			belong[v] = sum;
			num[sum] ++;
			if(u == v)
				break;
		}
	}
}
int main(){
	cin>>n>>m;
	for(int i = 1;i <= m;i ++){
		int u,v;
		cin>>u>>v;
		add(u,v);
	}
	for(int i = 1;i <= n;i ++)
		if(!dfn[i])
			tarjan(i);
	for(int i = 1;i <= m;i ++)
		if(belong[e[i].u] != belong[e[i].v])
			out[belong[e[i].u]] ++;
	for(int i = 1;i <= sum;i ++)
		if(out[i] == 0){
			res ++;
			ans = i;
		}
	if(res == 1)
		cout<<num[ans];
	else cout<<0;
	return 0;
}
```

#### kruskal 重构树

U92652

```cpp
#include <bits/stdc++.h>
using namespace std;
const int M = 24e5 + 20;
struct edge {
    int u, v, w, nxt;
}a[M], e[M];
int n, m, q, head[M], cnt, tot, d[M];
int fa[M];
int dep[M], f[M][25];
bool cmp(edge a, edge b) {
    return a.w < b.w;
}
void add(int u, int v, int w) {
    e[++ cnt] = {u, v, w, head[u]};
    head[u] = cnt;
}
void init() {
    for(int i = 1; i <= 2 * n; i ++)
        fa[i] = i;
}
int find(int x) {
    return (x == fa[x] ? x : fa[x] = find(fa[x]));
}
void kruskal() {
    sort(a + 1, a + m + 1, cmp);
    for(int i = 1; i <= m; i ++) {
        int fu = find(a[i].u);
        int fv = find(a[i].v);
        if(fu == fv)
            continue;
        tot ++;
        // cout << "%%%" << tot << '\n';
        fa[fu] = tot;
        fa[fv] = tot;
        d[tot] = a[i].w;
        add(tot, fu, 1);
        add(tot, fv, 1);
    }
}
void dfs(int u, int fa){
	dep[u] = dep[fa] + 1;
	f[u][0] = fa;
	for(int i = 1;(1 << i) <= dep[u];i ++)
		f[u][i] = f[f[u][i - 1]][i - 1];
	for(int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v;
        if(v == fa)
            continue;
        dfs(v, u);
    }
}
int lca(int x, int y) {
    if(dep[x] < dep[y])
        swap(x, y);
    int t = dep[x] - dep[y];
    for(int i = 21; i >= 0; i --)
        if(t & (1 << i))
            x = f[x][i];
    if(x == y)
        return d[x];
    for(int i = 21; i >= 0; i --)
        if(f[x][i] != f[y][i]) {
            x = f[x][i];
            y = f[y][i];
        }
    return d[f[x][0]];
}
int main() {
    ios::sync_with_stdio(false), cin.tie(0);
    cin >> n >> m >> q;
    tot = n;
    init();
    for(int i = 1; i <= m; i ++)
        cin >> a[i].u >> a[i].v >> a[i].w;
    kruskal();
    for(int i = 1; i <= 2 * n; i ++)
        if(fa[i] == i)
            dfs(i, 0);
    for(int i = 1; i <= q; i ++) {
        int x, y;
        cin >> x >> y;
        if(find(x) != find(y))
            cout << -1 << '\n';
        else cout << lca(x, y) << '\n';
        // cout << "%%% YSC && ZSC " << find(x) << " " << find(y) << '\n'; 
    }
    return 0;
}
```

#### 割点

P3388

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 2e4 + 20, M = 2e5 + 20;
struct edge {
    int u, v, nxt;
}e[M];
int n, m;
int cnt, head[N];
int low[N], dfn[N], tot;
set<int> ans;
void add(int u, int v) {
    e[++ cnt] = {u, v, head[u]};
    head[u] = cnt;
}
void tarjan(int u, int fa) {
    dfn[u] = low[u] = ++ tot;
    int son = 0;
    for(int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v;
        if(v == fa) continue;
        if(!dfn[v]) {
            son ++;
            tarjan(v, u);
            low[u] = min(low[u], low[v]);
            if(son > 1 && !fa || fa && low[v] >= dfn[u])
                ans.insert(u);
        }
        else low[u] = min(low[u], dfn[v]);
    }
}
int main() {
    cin >> n >> m;
    for(int i = 1; i <= m; i ++) {
        int u, v;
        cin >> u >> v;
        add(u, v), add(v, u);
    }
    for(int i = 1; i <= n; i++)
        if(!dfn[i])
            tarjan(i, 0);
    cout << ans.size() << '\n';
    for(int v : ans)
        cout << v << ' ';
    return 0;
}
```

#### 割边

P1656

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 170, M = 1e4 + 20;
struct edge {
    int u, v, nxt;
}e[M];
int n, m;
int cnt, head[N];
int low[N], dfn[N], tot;
bool b[M];
vector <pair <int, int> > ans; 
void add(int u, int v) {
    e[++ cnt] = {u, v, head[u]};
    head[u] = cnt;
}
void tarjan(int u, int fa) {
    dfn[u] = low[u] = ++ tot;
    for(int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v;
        if(v == fa) continue;
        if(!dfn[v]) {
            tarjan(v, u);
            low[u] = min(low[u], low[v]);
            if(low[v] > dfn[u])
                ans.push_back({min(u, v), max(u, v)});
        }
        else low[u] = min(low[u], dfn[v]);
    }
}
int main() {
    cin >> n >> m;
    for(int i = 1; i <= m; i ++) {
        int u, v;
        cin >> u >> v;
        add(u, v), add(v, u);
    }
    tarjan(1, 0);
    sort(ans.begin(), ans.end());
    for(auto i : ans)
        cout << i.first << " " << i.second << '\n';
    return 0;
}
```

#### 位运算技巧

![位运算技巧](https://cdn.luogu.com.cn/upload/image_hosting/vz03m4xz.png)
