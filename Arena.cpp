#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <array>
#include <random>
#include <list>
#include <chrono>
#include <omp.h>
#include <limits>
#include <algorithm>
#include <map>
#include <queue>
#include <thread>
#include <csignal>
using namespace std;
using namespace std::chrono;

constexpr int N{2};//Number of players
constexpr bool Debug_AI{false},Timeout{false};
constexpr int PIPE_READ{0},PIPE_WRITE{1};
constexpr double FirstTurnTime{1*(Timeout?1:10)},TimeLimit{0.1*(Timeout?1:10)};

bool stop{false};//Global flag to stop all arena threads when SIGTERM is received

struct vec{
    int x,y;
    inline int idx(const int W)const noexcept{
        return y*W+x;
    }
    inline bool valid(const int W)const noexcept{
        return x>=0 && y>=0 && x<W && y<W;
    }
    inline vec operator+(const vec &a)const noexcept{
        return vec{x+a.x,y+a.y};
    }
    inline vec operator-(const vec &a)const noexcept{
        return vec{x-a.x,y-a.y};
    }
    inline bool operator==(const vec &a)const noexcept{
        return x==a.x && y==a.y;
    }
};

inline ostream& operator<<(ostream &os,const vec &r){
    os << r.x << " " << r.y;
    return os;
}

const vector<string> DirStr{"N","NE","E","SE","S","SW","W","NW"};
constexpr array<vec,8> Dir{vec{0,-1},vec{1,-1},vec{1,0},vec{1,1},vec{0,1},vec{-1,1},vec{-1,0},vec{-1,-1}};
const map<string,int> DirToInt{{"N",0},{"NE",1},{"E",2},{"SE",3},{"S",4},{"SW",5},{"W",6},{"NW",7}};

struct state{
    vector<vec> P;
    vector<int> G;
    array<int,2> Score;
    inline int Occupant(const vec &r)const noexcept{
        for(int i=0;i<P.size();++i){
            if(r==P[i]){
                return i;
            }
        }
        return -1;
    }
    inline void Build(const int idx)noexcept{
        G[idx]=G[idx]<3?G[idx]+1:-1;
    }
};

enum action_type{MOVE=0,PUSH=1};
const map<string,action_type> StrToType{{"MOVE&BUILD",MOVE},{"PUSH&BUILD",PUSH}};

struct action{
    action_type type;
    int id;
    vec t,b;
    inline bool operator==(const action &a)const noexcept{
        return type==a.type && t==a.t && b==a.b && id==a.id;//Don't check for id because it doesn't really matter what pawn is pushing?
    }
};

inline string EmptyPipe(const int fd){
    int nbytes;
    if(ioctl(fd,FIONREAD,&nbytes)<0){
        throw(4);
    }
    string out;
    out.resize(nbytes);
    if(read(fd,&out[0],nbytes)<0){
        throw(4);
    }
    return out;
}

struct AI{
    int id,pid,outPipe,errPipe,inPipe,turnOfDeath;
    string name;
    inline void stop(const int turn=-1){
        if(alive()){
            kill(pid,SIGTERM);
            int status;
            waitpid(pid,&status,0);//It is necessary to read the exit code for the process to stop
            if(!WIFEXITED(status)){//If not exited normally try to "kill -9" the process
                kill(pid,SIGKILL);
            }
            turnOfDeath=turn;
        }
    }
    inline bool alive()const{
        return kill(pid,0)!=-1;//Check if process is still running
    }
    inline void Feed_Inputs(const string &inputs){
        if(write(inPipe,&inputs[0],inputs.size())!=inputs.size()){
            throw(5);
        }
    }
    inline ~AI(){
        close(errPipe);
        close(outPipe);
        close(inPipe);
        stop();
    }
};

void StartProcess(AI &Bot){
    int StdinPipe[2];
    int StdoutPipe[2];
    int StderrPipe[2];
    if(pipe(StdinPipe)<0){
        perror("allocating pipe for child input redirect");
    }
    if(pipe(StdoutPipe)<0){
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        perror("allocating pipe for child output redirect");
    }
    if(pipe(StderrPipe)<0){
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        perror("allocating pipe for child stderr redirect failed");
    }
    int nchild{fork()};
    if(nchild==0){//Child process
        if(dup2(StdinPipe[PIPE_READ],STDIN_FILENO)==-1){// redirect stdin
            perror("redirecting stdin");
            return;
        }
        if(dup2(StdoutPipe[PIPE_WRITE],STDOUT_FILENO)==-1){// redirect stdout
            perror("redirecting stdout");
            return;
        }
        if(dup2(StderrPipe[PIPE_WRITE],STDERR_FILENO)==-1){// redirect stderr
            perror("redirecting stderr");
            return;
        }
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        execl(Bot.name.c_str(),Bot.name.c_str(),(char*)NULL);//(char*)Null is really important
        //If you get past the previous line its an error
        perror("exec of the child process");
    }
    else if(nchild>0){//Parent process
        close(StdinPipe[PIPE_READ]);//Parent does not read from stdin of child
        close(StdoutPipe[PIPE_WRITE]);//Parent does not write to stdout of child
        close(StderrPipe[PIPE_WRITE]);//Parent does not write to stderr of child
        Bot.inPipe=StdinPipe[PIPE_WRITE];
        Bot.outPipe=StdoutPipe[PIPE_READ];
        Bot.errPipe=StderrPipe[PIPE_READ];
        Bot.pid=nchild;
    }
    else{//failed to create child
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        perror("Failed to create child process");
    }
}

inline bool IsValidMove(const state &S,const AI &Bot,const string &M){
    return count(M.begin(),M.end(),'\n')==1;
}

string GetMove(const state &S,AI &Bot,const int turn){
    pollfd outpoll{Bot.outPipe,POLLIN};
    time_point<system_clock> Start_Time{system_clock::now()};
    string out;
    while(static_cast<duration<double>>(system_clock::now()-Start_Time).count()<(turn==1?FirstTurnTime:TimeLimit) && !IsValidMove(S,Bot,out)){
        double TimeLeft{(turn==1?FirstTurnTime:TimeLimit)-static_cast<duration<double>>(system_clock::now()-Start_Time).count()};
        if(poll(&outpoll,1,TimeLeft)){
            out+=EmptyPipe(Bot.outPipe);
        }
    }
    return out;
}

inline bool Has_Won(const array<AI,N> &Bot,const int idx)noexcept{
    if(!Bot[idx].alive()){
        return false;
    }
    for(int i=0;i<N;++i){
        if(i!=idx && Bot[i].alive()){
            return false;
        }
    }
    return true;
}

inline bool All_Dead(const array<AI,N> &Bot)noexcept{
    for(const AI &b:Bot){
        if(b.alive()){
            return false;
        }
    }
    return true;
}

action StringToAction(const state &S,const string &M_Str,const int playerId){
    action mv;
    stringstream ss(M_Str);
    string type;
    if(!(ss >> type)){
        throw(1);
    }
    if(type=="MOVE&BUILD" || type=="PUSH&BUILD"){
        mv.type=StrToType.at(type);
        if(!(ss >> mv.id)){
            throw(1);
        }
        mv.id+=playerId*2;
        string dir1,dir2;
        if(!(ss >> dir1 >> dir2)){
            throw(1);
        }
        const int dir1_idx{DirToInt.at(dir1)},dir2_idx{DirToInt.at(dir2)};
        if(type=="PUSH&BUILD"){
            const array<int,3> Possible_Pushes{dir1_idx==0?7:dir1_idx-1,dir1_idx,dir1_idx==7?0:dir1_idx+1};
            if(find(Possible_Pushes.begin(),Possible_Pushes.end(),dir2_idx)==Possible_Pushes.end()){
                cerr << "Push in illegal angle" << endl;
                throw(1);
            }
        }
        mv.t=S.P[mv.id]+Dir[dir1_idx];
        mv.b=mv.t+Dir[dir2_idx];
    }
    else if(type=="ACCEPT-DEFEAT"){
        throw(6);
    }
    else{
        cerr << "Invalid action: " << M_Str << endl;
        throw(3);
    }
    return mv;
}

inline int Dist(const vec &a,const vec &b)noexcept{
    return max(abs(a.x-b.x),abs(a.y-b.y));
}

inline bool Visible(const state &S,const int playerId,const vec &r)noexcept{
    for(int i=0;i<2;++i){
        if(Dist(S.P[playerId*2+i],r)<=1){
            return true;
        }
    }
    return false;
}

void Simulate(state &S,const action &mv,const int playerId){
    const int W{static_cast<int>(sqrt(S.G.size()))};
    if(!mv.t.valid(W) || !mv.b.valid(W)){
        cerr << "Tried to take action out side map t: " << mv.t << " b: " << mv.b << endl;
        throw(3);
    }
    if(S.G[mv.t.idx(W)]==-1 || S.G[mv.b.idx(W)]==-1){
        cerr << "Tried to move or push on unplayable square" << endl;
        throw(3);
    }
    if(mv.type==MOVE){
        const int max_height{S.G[S.P[mv.id].idx(W)]+1};
        const bool move_occupied{S.Occupant(mv.t)!=-1};
        const bool build_occupied{(S.Occupant(mv.b)!=-1 && S.Occupant(mv.b)!=mv.id && Visible(S,playerId,mv.b))};\
        const bool move_too_high{S.G[mv.t.idx(W)]>max_height};
        if(move_occupied || build_occupied || move_too_high){
            if(move_occupied){
                cerr << "Tried to move to a location occupied by another pawn" << endl;
            }
            if(build_occupied){
                cerr << "Tried to build on a location occupied by another pawn" << endl;
            }
            if(move_too_high){
                cerr << "Tried to move to a location that was too high compared to the starting point" << endl;
            }
            throw(3);
        }
        S.P[mv.id]=mv.t;
        if(S.G[mv.t.idx(W)]==3){
            ++S.Score[playerId];
        }
        if(S.Occupant(mv.b)==-1){
            S.Build(mv.b.idx(W));
        }
    }
    else{
        const int occupant{S.Occupant(mv.t)};
        const int max_height{S.G[mv.b.idx(W)]+1};
        const bool noone_to_push{occupant==-1};
        const bool push_too_high{S.G[mv.b.idx(W)]>max_height};
        const bool push_on_known_pawn{(S.Occupant(mv.b)!=-1 && Visible(S,playerId,mv.b))};
        if(noone_to_push || push_too_high || push_on_known_pawn){
            if(noone_to_push){
                cerr << "Tried to push a location without a pawn" << endl;
            }
            if(push_too_high){
                cerr << "Tried to push someone to a location that was too high for them to move to" << endl;
            }
            if(push_on_known_pawn){
                cerr << "Tried to push on a visible pawn" << endl;
            }
            throw(3);
        }
        if(S.Occupant(mv.b)==-1){
            S.P[occupant]=mv.b;
            S.Build(mv.t.idx(W));
        }
    }
}

int Play_Game(const array<string,N> &Bot_Names,state S){
    array<AI,N> Bot;
    const int W{static_cast<int>(sqrt(S.G.size()))};
    for(int i=0;i<N;++i){
        Bot[i].id=i;
        Bot[i].name=Bot_Names[i];
        StartProcess(Bot[i]);
        stringstream ss;
        ss << W << endl;
        ss << 2 << endl;
        Bot[i].Feed_Inputs(ss.str());
    }
    int turn{0};
    while(++turn>0 && !stop){
        //cerr << turn << endl;
        for(int id=0;id<N;++id){
            if(Bot[id].alive()){
                stringstream ss;
                for(int i=0;i<W;++i){
                    for(int j=0;j<W;++j){
                        if(S.G[i*W+j]==-1){
                            ss << '.';
                        }
                        else{
                            ss << S.G[i*W+j];
                        }
                    }
                    ss << endl;
                }
                for(int i=0;i<2;++i){
                    ss << S.P[id*2+i] << endl;
                }
                for(int i=0;i<2;++i){
                    if(Visible(S,id,S.P[(!id)*2+i])){
                        ss << S.P[(!id)*2+i] << endl;
                    }
                    else{
                        ss << vec{-1,-1} << endl;
                    }
                }
                ss << 0 << endl;//number of legal moves I'm passing
                //cerr << ss.str();
                try{
                    Bot[id].Feed_Inputs(ss.str());
                    string out=GetMove(S,Bot[id],turn);
                    //cerr << id << " " << out << endl;
                    string err_str{EmptyPipe(Bot[id].errPipe)};
                    if(Debug_AI){
                        ofstream err_out("log.txt",ios::app);
                        err_out << err_str << endl;
                    }
                    const action mv=StringToAction(S,out,id);
                    Simulate(S,mv,id);
                }
                catch(int ex){
                    if(ex==1){//Timeout
                        cerr << "Loss by Timeout of AI " << Bot[id].id << " name: " << Bot[id].name << endl;
                    }
                    else if(ex==3){
                        cerr << "Invalid move from AI " << Bot[id].id << " name: " << Bot[id].name << endl;
                    }
                    else if(ex==4){
                        cerr << "Error emptying pipe of AI " << Bot[id].name << endl;
                    }
                    else if(ex==5){
                        cerr << "AI " << Bot[id].name << " died before being able to give it inputs" << endl;
                    }
                    Bot[id].stop(turn);
                }
            }
            else if(S.Score[id]<S.Score[!id]){
                return !id;//Other guy has won cause dead guy's score can't increase
            }
        }
        if(All_Dead(Bot) || turn==200){
            return S.Score[0]>S.Score[1]?0:S.Score[1]>S.Score[0]?1:-1;
        }
    }
    return -2;
}

int Components(const vector<int> &G){
    const int W{static_cast<int>(sqrt(G.size()))};
    vector<char> visited(G.size(),false);
    queue<vec> bfs_queue;
    int components{0};
    for(int i=0;i<G.size();++i){
        if(!visited[i]){
            ++components;
            visited[i]=true;
            bfs_queue.push(vec{i%W,i/W});
            while(!bfs_queue.empty()){
                const vec r=bfs_queue.front();
                bfs_queue.pop();
                for(const vec &d:Dir){
                    const vec t=r+d;
                    if(t.valid(W) && !visited[t.idx(W)] && G[t.idx(W)]!=-1){
                        bfs_queue.push(t);
                        visited[t.idx(W)]=true;
                    }
                }
            } 
        }
    }
    return components;
}

vector<int> RandomMap(default_random_engine &generator){
    constexpr int W{6};
    vector<int> G(pow(W,2),-1);
    uniform_int_distribution<int> Cells_Distrib(25,34),X_Distrib(0,5);
    const int Desired_Cells{Cells_Distrib(generator)};
    int N_Cells{0};
    while(N_Cells<Desired_Cells || Components(G)>1){
        const vec r{X_Distrib(generator),X_Distrib(generator)};
        const vec mirror{6-1-r.x,r.y};
        G[r.idx(W)]=0;
        G[mirror.idx(W)]=0;
        N_Cells+=2;
    }
    return G;
}

inline bool Spawn_Taken(const vector<vec> &Spawn,const vec &r){
    for(const vec &s:Spawn){
        if(r==s){
            return true;
        }
    }
    return false;
}

int Play_Round(array<string,N> Bot_Names){
    default_random_engine generator(system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> Swap_Distrib(0,1);
    const bool player_swap{Swap_Distrib(generator)==1};
    if(player_swap){
        swap(Bot_Names[0],Bot_Names[1]);
    }
    
    state S;
    const vector<int> SquareMap{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const vector<int> DiamondMap{-1,-1,-1,0,-1,-1,-1,-1,-1,0,0,0,-1,-1,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,-1,0,0,0,0,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,-1,0,-1,-1,-1};
    const vector<vector<int>> Maps{SquareMap,DiamondMap,RandomMap(generator)};
    uniform_int_distribution<int> Map_Distrib(0,2);
    S.G=Maps[Map_Distrib(generator)];
    const int W{static_cast<int>(sqrt(S.G.size()))};
    uniform_int_distribution<int> X_Distrib(0,W-1);
    vector<vec> Spawn(4,vec{-1,-1});
    for(int i=0;i<4;++i){
        vec r;
        do{
            r=vec{X_Distrib(generator),X_Distrib(generator)};
        }while(Spawn_Taken(Spawn,r) || S.G[r.idx(W)]==-1);
        Spawn[i]=r;
    }
    S.P.resize(4);
    fill(S.Score.begin(),S.Score.end(),0);
    for(int j=0;j<4;++j){
        S.P[j]=Spawn[j];
    }
    
    int winner{Play_Game(Bot_Names,S)};
    if(player_swap){
        return winner==-1?-1:winner==0?1:0;
    }
    else{
        return winner;
    }
}

void StopArena(const int signum){
    stop=true;
}

int main(int argc,char **argv){
    if(argc<3){
        cerr << "Program takes 2 inputs, the names of the AIs fighting each other" << endl;
        return 0;
    }
    int N_Threads{1};
    if(argc>=4){//Optional N_Threads parameter
        N_Threads=min(2*omp_get_num_procs(),max(1,atoi(argv[3])));
        cerr << "Running " << N_Threads << " arena threads" << endl;
    }
    array<string,N> Bot_Names;
    for(int i=0;i<2;++i){
        Bot_Names[i]=argv[i+1];
    }
    cout << "Testing AI " << Bot_Names[0];
    for(int i=1;i<N;++i){
        cerr << " vs " << Bot_Names[i];
    }
    cerr << endl;
    for(int i=0;i<N;++i){//Check that AI binaries are present
        ifstream Test{Bot_Names[i].c_str()};
        if(!Test){
            cerr << Bot_Names[i] << " couldn't be found" << endl;
            return 0;
        }
        Test.close();
    }
    signal(SIGTERM,StopArena);//Register SIGTERM signal handler so the arena can cleanup when you kill it
    signal(SIGPIPE,SIG_IGN);//Ignore SIGPIPE to avoid the arena crashing when an AI crashes
    int games{0},draws{0};
    array<double,2> points{0,0};
    #pragma omp parallel num_threads(N_Threads) shared(games,points,Bot_Names)
    while(!stop){
        int winner{Play_Round(Bot_Names)};
        if(winner==-1){//Draw
            #pragma omp atomic
            ++draws;
            #pragma omp atomic
            points[0]+=0.5;
            #pragma omp atomic
            points[1]+=0.5;
        }
        else{//Win
            #pragma omp atomic
            points[winner]+=1;
        }
        #pragma omp atomic
        games+=1;
        double p{static_cast<double>(points[0])/games};
        double sigma{sqrt(p*(1-p)/games)};
        double better{0.5+0.5*erf((p-0.5)/(sqrt(2)*sigma))};
        #pragma omp critical
        cout << "Wins:" << setprecision(4) << 100*p << "+-" << 100*sigma << "% Rounds:" << games << " Draws:" << draws << " " << better*100 << "% chance that " << Bot_Names[0] << " is better" << endl;
    }
}