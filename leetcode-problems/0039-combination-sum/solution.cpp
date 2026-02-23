#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> path;
        dfs(0, target, candidates, path, res);
        return res;
    }
    
private:
    void dfs(int i, int current_target, vector<int>& candidates, vector<int>& path, vector<vector<int>>& res) {
        // Base Case 1: Valid combination found
        if (current_target == 0) {
            res.push_back(path);
            return;
        }
        
        // Base Case 2: Invalid path (overshot or out of bounds)
        if (current_target < 0 || i >= candidates.size()) {
            return;
        }
        
        // Decision 1: Pick the current candidate
        path.push_back(candidates[i]);
        // Stay at index 'i' to allow infinite reuse
        dfs(i, current_target - candidates[i], candidates, path, res);
        
        // Decision 2: Skip the candidate
        // Backtrack by removing the element we just added
        path.pop_back();
        // Move to the next index
        dfs(i + 1, current_target, candidates, path, res);
    }
};
