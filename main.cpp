#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <fstream>
#include <getopt.h>

#include "base.hpp"
#include "utils.hpp"
#include "bbtsvd.hpp"
#include "client.hpp"

extern int optopt;
extern char *optarg;

static struct option long_options[] = 
{
   {"listen", required_argument, NULL, 'l'},
   {"port", required_argument, NULL, 'p'},
   {"drow", required_argument, NULL, 'm'},
   {"dcol", required_argument, NULL, 'n'},
   {"dpath", required_argument, NULL, 'd'},
   {"logdir", required_argument, NULL, 'g'},
   {"topk", required_argument, NULL, 'k'},
   {"mode", required_argument, NULL, 'o'},
   {"nclient", required_argument, NULL, 't'},
   {"cid", required_argument, NULL, 'c'},
   {"evaluate", required_argument, NULL, 'e'},
   {"ismemmap", required_argument, NULL, 'f'},
   {"seed", required_argument, NULL, 's'},
   {"opt", required_argument, NULL, 'a'},
   {"help", no_argument, NULL, 'h'},
};

/*

apt install -y iproute2

sudo sysctl -w net.ipv4.tcp_rmem='40960 873800 62914560'
sudo sysctl -w net.ipv4.tcp_wmem='40960 873800 62914560'

*/

int main(int argc, char** argv)
{
   
   int comm=0, index=0, m, n, port, topk, mode, num_client, cid, opt;
   string listen, dpath, logdir;
   bool evaluate, is_memmap;
   unsigned int seed;
   
   // If you have 8 CPU cores and your machine has no super-threading:
   // mkl_set_num_threads(omp_get_max_threads() - 1);
   // omp_set_num_threads(omp_get_max_threads() - 1);
   mkl_set_num_threads(omp_get_max_threads());
   omp_set_num_threads(omp_get_max_threads());
   
   while(EOF != (comm = getopt_long(argc, argv, "l:p:m:n:k:o:d:g:t:c:e:f:s:a:h", long_options, &index)))
   {
      switch(comm){
         case 'l':
            listen = optarg; cout << "Host=" << listen << endl;
         case 'p':
            port = atoi(optarg); cout << "Port=" << port << endl; break;
         case 'm':
            m = atoi(optarg); cout << "DataRow=" << m << endl; break;
         case 'n':
            n = atoi(optarg); cout << "DataCol=" << n << endl; break;
         case 'k':
            topk = atoi(optarg); cout << "TopK=" << topk << endl; break;
         case 'o':
            mode = atoi(optarg); cout << "Mode=" << mode << endl; break;
         case 'd':
            dpath = optarg; cout << "DataPath=" << dpath << endl; break;
         case 'g':
            logdir = optarg; cout << "LogDir=" << logdir << endl; break;
         case 't':
            num_client = atoi(optarg); cout << "NumClient=" << num_client << endl; break;
         case 'c':
            cid = atoi(optarg); cout << "CID=" << cid << endl; break;
         case 'e':
            evaluate = atoi(optarg) == 1 ? true : false; cout << "evaluate=" << evaluate << endl; break;
         case 'f':
            is_memmap = atoi(optarg) == 1 ? true : false; cout << "is_memmap=" << is_memmap << endl; break;
         case 's':
            seed = atoi(optarg); cout << "Seed=" << seed << endl; break;
         case 'a':
            opt = atoi(optarg); cout << "Opt=" << opt << endl; break;
         case 'h':
            cout << "Params Helper" << endl;
            cout << " -l --listen   : base server IP\n";
            cout << " -p --port     : base server port\n";
            cout << " -m --drow     : # data rows\n";
            cout << " -n --dcol     : # data cols\n";
            cout << " -d --dpath    : data path\n";
            cout << " -k --topk     : top-k of PCA and LSA\n";
            cout << " -o --mode     : 0=SVD, 1=LR, 2=PCA\n";
            cout << " -t --nclienr  : number of client\n";
            cout << " -c --cid      : my client id\n";
            return 0;
      }
   }
   
   DecentralizedClient client = DecentralizedClient(cid, num_client, mode, topk, m, n, port, listen, dpath, is_memmap, evaluate, logdir, seed, opt);
   
   return 0;
}