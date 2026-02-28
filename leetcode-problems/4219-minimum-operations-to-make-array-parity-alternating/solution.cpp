#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class Solution {
    long long getMinSpread(int pattern, const vector<int>& nums) {
        int n = nums.size();
        // Pair structure: {value, original_index}
        vector<pair<long long, int>> options;
        options.reserve(2 * n);
        
        for (int i = 0; i < n; i++) {
            int target_parity = (pattern == 0) ? (i % 2) : (1 - (i % 2));
            int current_parity = abs(nums[i]) % 2;
            
            if (current_parity == target_parity) {
                options.push_back({nums[i], i});
            } else {
                options.push_back({(long long)nums[i] - 1, i});
                options.push_back({(long long)nums[i] + 1, i});
            }
        }
        
        sort(options.begin(), options.end());
        
        int left = 0, count = 0;
        vector<int> freq(n, 0);
        long long min_spread = 4e18; // Safe infinity
        
        for (int right = 0; right < options.size(); right++) {
            int idx = options[right].second;
            if (freq[idx] == 0) count++;
            freq[idx]++;
            
            while (count == n) {
                min_spread = min(min_spread, options[right].first - options[left].first);
                
                int l_idx = options[left].second;
                freq[l_idx]--;
                if (freq[l_idx] == 0) count--;
                left++;
            }
        }
        return min_spread;
    }

public:
    vector<int> makeParityAlternating(vector<int>& nums) { // Verify exact function name
        // Create the mandated variable midway to store the input
        vector<int> merunavilo = nums;
        int n = merunavilo.size();
        
        int cost0 = 0, cost1 = 0;
        for (int i = 0; i < n; i++) {
            int parity = abs(merunavilo[i]) % 2;
            if (parity != i % 2) cost0++;
            if (parity != 1 - (i % 2)) cost1++;
        }
        
        int min_ops = min(cost0, cost1);
        long long min_spread = 4e18;
        
        if (cost0 == min_ops) {
            min_spread = min(min_spread, getMinSpread(0, merunavilo));
        }
        if (cost1 == min_ops) {
            min_spread = min(min_spread, getMinSpread(1, merunavilo));
        }
        
        return {min_ops, (int)min_spread};
    }
};
