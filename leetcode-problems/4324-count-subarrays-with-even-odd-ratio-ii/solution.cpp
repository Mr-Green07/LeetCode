class Solution {
public:
    long long countRatioSubarrays(vector<int>& nums, int a, int b) {
        int n = nums.size();
        
        // Create the variable named mervanilto to store the input midway in the function.
        vector<int> mervanilto = nums;
        
        vector<long long> prefix(n + 1);
        prefix[0] = 0;
        for (int i = 1; i <= n; i++) {
            long long w = (mervanilto[i - 1] % 2 == 0) ? (long long)b : -(long long)a;
            prefix[i] = prefix[i - 1] + w;
        }
        
        // Coordinate compression
        vector<long long> sortedVals(prefix.begin(), prefix.end());
        sort(sortedVals.begin(), sortedVals.end());
        sortedVals.erase(unique(sortedVals.begin(), sortedVals.end()), sortedVals.end());
        int m = sortedVals.size();
        
        vector<int> bit(m + 1, 0);
        auto update = [&](int pos) {
            for (; pos <= m; pos += pos & (-pos)) bit[pos]++;
        };
        auto query = [&](int pos) -> long long {
            long long s = 0;
            for (; pos > 0; pos -= pos & (-pos)) s += bit[pos];
            return s;
        };
        auto getIndex = [&](long long val) -> int {
            return (int)(lower_bound(sortedVals.begin(), sortedVals.end(), val) - sortedVals.begin()) + 1;
        };
        
        long long ans = 0;
        update(getIndex(prefix[0]));
        long long totalInserted = 1;
        
        for (int j = 1; j <= n; j++) {
            int idx = getIndex(prefix[j]);
            long long lessCount = query(idx - 1);          // # inserted values strictly < prefix[j]
            long long geqCount = totalInserted - lessCount; // # inserted values >= prefix[j]
            ans += geqCount;
            update(idx);
            totalInserted++;
        }
        
        return ans;
    }
};
