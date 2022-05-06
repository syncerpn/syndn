#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

void train_classifier_slimmable(char *datacfg, char *cfgfile, char *weightfile, char* modules, int *gpus, int ngpus, int clear, int info)
{

    list *options = read_data_cfg(datacfg);
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *info_directory = option_find_str(options, "info", "/info/");
    char *label_path = option_find_str(options, "labels", 0);

    int classes = option_find_int(options, "classes", 2);

    char *base = basecfg(cfgfile);

    char infofile_name[256];
    sprintf(infofile_name, "%s/%s_%u.info", info_directory, base, (unsigned)time(NULL));

    fprintf(stderr, "%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network*));

    int i;
    for(i = 0; i < ngpus; ++i){
        cuda_set_device(gpus[i]);
        nets[i] = load_network(cfgfile, weightfile, modules, clear, 0);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);

    int imgs = net->batch * net->subdivisions * ngpus;

    char **labels = get_labels(label_path);
    list *plist = get_paths(train_list);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    double time;

    load_args args = get_base_args(net);

    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    fprintf(stderr, "%d %d\n", args.min, args.max);

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

    data train, buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;

    summarize_data_augmentation_options(args);

    FILE* infofile = 0;
    if (info) {
        infofile = fopen(infofile_name, "wb");
        save_training_info(infofile, net, 1, N);
    }
    fprintf(stderr, "Learning Rate: %g, Decay: %g\n", net->learning_rate, net->decay);

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
            printf("Resizing\n");
            int dim = (rand() % 11 + 4) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
            printf("%d %d\n", args.min, args.max);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        time = what_time_is_it_now();
        float loss = 0;
        if(ngpus == 1){
            loss = train_network_slimmable(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%7ld: ", get_current_batch(net));
        int il;
        for (il = 0; il < net->n_loss; ++il) {
            printf("%9.6f (%d-%s)  ", net->sub_loss[il], net->sub_loss_id[il], get_layer_string(net->layers[net->sub_loss_id[il]].type));
        }
        printf("%9.6f (avg)  %8.6f (lr)  %5.2f (s)  %10ld (imgs)  %6.2f (epochs)\n", avg_loss, get_current_rate(net), what_time_is_it_now()-time, get_current_batch(net)*imgs, (float)(*net->seen)/N);
        
        if (info) save_training_info(infofile, net, 0, N);

        free_data(train);

        if (get_current_batch(net) % 1000 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
            if (info) {
                //save info file frequently; close then reopen as appending
                fclose(infofile);
                infofile = fopen(infofile_name, "ab");
                fprintf(stderr, "Save training info to %s\n", infofile_name);
            }
        }

        if (get_current_batch(net) % 5000 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s_%lu.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);
    if (info) {
        fclose(infofile);
        fprintf(stderr, "Training info: %s\n", infofile_name);
    }

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void run_classifier_slimmable(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);

    char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
    char *data = find_char_2arg(argc, argv, "-d", "--data", 0);
    char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
    char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
    
    if(0==strcmp(argv[2], "train")) {
        int info = find_arg(argc, argv, "-info");
        int clear = find_arg(argc, argv, "-clear");
        train_classifier_slimmable(data, cfg, weights, modules, gpus, ngpus, clear, info);
    }
}