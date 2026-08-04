class Solution {
public:
    vector<int> findMissingElements(vector<int>& nums) {
        int high = *max_element(nums.begin(), nums.end());
        int low = *min_element(nums.begin(), nums.end());

        unordered_set<int> given(nums.begin(), nums.end());
        vector<int> result;
        for(int i = low; i<= high; i++){
            if(given.find(i) == given.end()){
                result.push_back(i);
            }
        }
        return result;
        

    }
};
