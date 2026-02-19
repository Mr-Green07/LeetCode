
using namespace std;

class Solution {
public:
    int minTaps(int n, vector<int>& ranges) {
        vector<int> max_reach(n + 1, 0);
        
        // 1. Build the intervals
        for (int i = 0; i <= n; ++i) {
            if (ranges[i] == 0) continue;
            int left = max(0, i - ranges[i]);
            int right = min(n, i + ranges[i]);
            max_reach[left] = max(max_reach[left], right);
        }
        
        int taps = 0;
        int curr_end = 0;
        int next_end = 0;
        
        // 2. Greedily traverse the garden
        for (int i = 0; i < n; ++i) {
            next_end = max(next_end, max_reach[i]);
            
            // Forced to open a new tap
            if (i == curr_end) {
                // If we can't extend further, it's impossible
                if (next_end <= i) return -1;
                
                taps++;
                curr_end = next_end;
            }
        }
        
        return taps;
    }
};
