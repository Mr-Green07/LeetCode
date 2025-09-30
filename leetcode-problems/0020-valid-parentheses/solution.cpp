class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        unordered_map<char, char> mapping ={
            {')', '('},
            {'}', '{'},
            {']', '['}
        };
        for(char c:s){
            if (mapping.count(c)){
                char top = st.empty()? '#' : st.top();
                if(!st.empty()) st.pop();
                if(top != mapping[c]){
                    return false;
                }
            } else{
                st.push(c);
            }
        }
        return st.empty();
    }
};
