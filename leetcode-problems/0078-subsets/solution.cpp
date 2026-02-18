class Solution {
public:
    void solve(vector<int>& nums, int index, vector<int> temp, vector<vector<int>> &res){
        if(index == nums.size()){
            res.push_back(temp);
            return ;
        }
        temp.push_back(nums[index]);
        solve(nums, index+1, temp, res);
        
        temp.pop_back();
        solve(nums, index+1, temp, res);
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> temp;
        
        int n = nums.size();
        int i =0;
        
        solve(nums, i, temp, res);
        

        return res;
    }
};
