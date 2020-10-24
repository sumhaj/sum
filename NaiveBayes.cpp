#include <iostream>
#include <string>
#include <bits/stdc++.h> 
#include <unordered_map>
using namespace std;



class words{
    public:
    long long int pos_count=0;
    long long int neg_count=0;
    double pos_prob=0;
    double neg_prob=0;
};



int main(){
    
    bool pos;
    // each word from document will have an object which keeps track of count and probability
    unordered_map<string,words> wrd;

    long long int tpw=0,tnw=0,tnuw=0,tpuw=0;

    //open a file stream
    fstream file("trainpos 100.txt");

    if (!file.is_open()){
        cout<<"error, file not open";
        return 0;
    }
    else{
        file.seekp(0,ios::end);
        size_t size = file.tellg();
            if( size == 0)
            {
                cout << "File is empty\n";
            }
    }
    file.seekg(0, ios::beg);
    string word;

    //reading words from file and counting pos count, neg count, total pos and neg words and unique pos and neg words
    while(file >> word){
        if(word=="<pos>"){                          
            pos=true;
        }
        else if(word=="<neg>"){
            pos=false;
        }
        else{
            if(!wrd.count(word)){
                if(pos==true){
                    wrd[word].pos_count=1;
                    tpuw++;
                    tpw++;
                }
                else{
                    wrd[word].neg_count=1;
                    tnuw++;
                    tnw++;
                }
            }
            else{
                 if(pos==true){
                    wrd[word].pos_count++;
                        if(wrd[word].pos_count==1){
                            tpuw++;
                        }
                    tpw++;
                }
                else{
                    wrd[word].neg_count++;
                    if(wrd[word].neg_count==1){
                            tnuw++;
                        }
                    tnw++;
                }
            }
        }
    }
    
 
    //traversing through hash map to calculate pos and neg probability of each word
    for(auto &w : wrd){

        // applying  laplace smoothing

        long long int num1=w.second.pos_count+1;
        long long int den1=tpw+tpuw;
        double pos_probability=(double)num1/(double)den1;
        w.second.pos_prob=log10(pos_probability);

        long long int num2=w.second.neg_count+1;
        long long int den2=tnw+tnuw;
        double neg_probability=(double) num2/(double) den2;
        w.second.neg_prob=log10(neg_probability);
    }

    //taking input string, break words-> calculate their probability,sum them upto to get the probability of the sentence being pos or neg
    string input_str;
    string str;
    double posProb=log10(0.5);
    double negProb=log10(0.5);
    getline(cin,input_str);
        if(input_str==""){
            cout<<"No Input"<<endl;
            return 0;
        }
    stringstream s(input_str);


    while(s >> str){
        posProb+= wrd[str].pos_prob;
        negProb+= wrd[str].neg_prob;
    }
    cout<<posProb<<endl<<negProb<<endl;
}