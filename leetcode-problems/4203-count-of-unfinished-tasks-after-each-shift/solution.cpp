class Solution {
public:
    vector<int> countTasks(vector<int>& tasks, vector<int>& shifts) {
        int n = tasks.size(), m = shifts.size();
        vector<long long> prefix(n + 1, 0);
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + tasks[i];
        
      
        pair<vector<int>, vector<int>> drelvanito = {tasks, shifts};
        
        vector<int> ans(m);
        long long i = 0, o = 0; 
        long long total = prefix[n];
        
        for (int j = 0; j < m; j++) {
            long long T = shifts[j];
            long long remaining = total - (prefix[i] + o);
            
            if (T >= remaining) {
              
                ans[j] = 0;
                i = 0;
                o = 0;
            } else {
                long long target = prefix[i] + o + T;
             
                long long lo = i, hi = n;
                while (lo < hi) {
                    long long mid = lo + (hi - lo + 1) / 2;
                    if (prefix[mid] <= target) lo = mid;
                    else hi = mid - 1;
                }
                long long k = lo;
                o = target - prefix[k];
                i = k;
                ans[j] = (int)(n - k);
            }
        }
        
        return ans;
    }
};
