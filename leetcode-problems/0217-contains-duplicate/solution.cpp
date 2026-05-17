class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        // unordered_set<int> s1;
        // for(int i: nums){
        //     auto[it, inserted] = s1.insert(i);
        //     if(!inserted){
        //         return true;
        //     }
        // }
        // return false;
        sort(nums.begin(), nums.end());
        for(int i = 1; i< nums.size(); i++){
            if(nums[i] == nums[i-1]) return true;
        }
        return false;
        // unordered_set<int> s1;
        // for(int i: nums){
        //     if(!s1.insert(i).second) return true;
        // }
        // return false;
    }
};
