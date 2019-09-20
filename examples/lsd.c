#include <math.h>
#include "darknet.h"

void slerp(float *start, float *end, float s, int n, float *out)
{
    float omega = acos(dot_cpu(n, start, 1, end, 1));
    float so = sin(omega);
    fill_cpu(n, 0, out, 1);
    axpy_cpu(n, sin((1-s)*omega)/so, start, 1, out, 1);
    axpy_cpu(n, sin(s*omega)/so, end, 1, out, 1);

    float mag = mag_array(out, n);
    scale_array(out, n, 1./mag);
}

image random_unit_vector_image(int w, int h, int c)
{
    image im = make_image(w, h, c);
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        im.data[i] = rand_normal();
    }
    float mag = mag_array(im.data, im.w*im.h*im.c);
    scale_array(im.data, im.w*im.h*im.c, 1./mag);
    return im;
}

void inter_dcgan(char *cfgfile, char *weightfile, char* modules)
{
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    clock_t time;
    char buff[256];
    char *input = buff;
    int i, imlayer = 0;

    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = i;
            printf("%d\n", i);
            break;
        }
    }
    image start = random_unit_vector_image(net->w, net->h, net->c);
    image end = random_unit_vector_image(net->w, net->h, net->c);
        image im = make_image(net->w, net->h, net->c);
        image orig = copy_image(start);

    int c = 0;
    int count = 0;
    int max_count = 15;
    while(1){
        ++c;
        
        if(count == max_count){
            count = 0;
            free_image(start);
            start = end;
            end = random_unit_vector_image(net->w, net->h, net->c);
            if(c > 300){
                end = orig;
            }
            if(c>300 + max_count) return;
        }
        ++count;

        slerp(start.data, end.data, (float)count / max_count, im.w*im.h*im.c, im.data);

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);

        normalize_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));

        sprintf(buff, "out%05d", c);
        save_image(out, "out");
        save_image(out, buff);
        show_image(out, "out", 0);
    }
}

void test_dcgan(char *cfgfile, char *weightfile, char* modules)
{
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    clock_t time;
    char buff[256];
    char *input = buff;
    int imlayer = 0;

    imlayer = net->n-1;

    while(1){
        image im = make_image(net->w, net->h, net->c);
        int i;
        for(i = 0; i < im.w*im.h*im.c; ++i){
            im.data[i] = rand_normal();
        }

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);

        normalize_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");
        show_image(out, "out", 0);

        free_image(im);
    }
}

void set_network_alpha_beta(network *net, float alpha, float beta)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == SHORTCUT){
            net->layers[i].alpha = alpha;
            net->layers[i].beta = beta;
        }
    }
}

void train_prog(char *cfg, char *weight, char* modules, char *acfg, char *aweight, char* amodules, int clear, int display, char *train_images, int maxbatch)
{
    char *backup_directory = "/home/pjreddie/backup/";
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network *gnet = load_network(cfg, weight, modules, clear, 0);
    network *anet = load_network(acfg, aweight, amodules, clear, 0);

    int i, j, k;
    layer imlayer = gnet->layers[gnet->n-1];

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", gnet->learning_rate, gnet->momentum, gnet->decay);
    int imgs = gnet->batch*gnet->subdivisions;
    i = *gnet->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(anet);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;
    args.threads=16;
    args.classes = 1;
    char *ls[2] = {"imagenet", "zzzzzzzz"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    gnet->train = 1;
    anet->train = 1;

    int x_size = gnet->inputs*gnet->batch;
    int y_size = gnet->truths*gnet->batch;
    float *imerror = cuda_make_array(0, y_size);

    float aloss_avg = -1;

    if (maxbatch == 0) maxbatch = gnet->max_batches;
    while (get_current_batch(gnet) < maxbatch) {
        {
            int cb = get_current_batch(gnet);
            float alpha = (float) cb / (maxbatch/2);
            if(alpha > 1) alpha = 1;
            float beta = 1 - alpha;
            printf("%f %f\n", alpha, beta);
            set_network_alpha_beta(gnet, alpha, beta);
            set_network_alpha_beta(anet, beta, alpha);
        }

        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gen = copy_data(train);
        for (j = 0; j < imgs; ++j) {
            train.y.vals[j][0] = 1;
            gen.y.vals[j][0] = 0;
        }
        time=clock();

        for (j = 0; j < gnet->subdivisions; ++j) {
            get_next_batch(train, gnet->batch, j*gnet->batch, gnet->truth, 0);
            int z;
            for(z = 0; z < x_size; ++z){
                gnet->input[z] = rand_normal();
            }

            *gnet->seen += gnet->batch;
            forward_network(gnet);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            fill_cpu(anet->truths*anet->batch, 1, anet->truth, 1);
            copy_cpu(anet->inputs*anet->batch, imlayer.output, 1, anet->input, 1);
            anet->delta_gpu = imerror;
            forward_network(anet);
            backward_network(anet);

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
            scal_gpu(imlayer.outputs*imlayer.batch, 0, gnet->layers[gnet->n-1].delta_gpu, 1);

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);

            backward_network(gnet);

            for(k = 0; k < gnet->batch; ++k){
                int index = j*gnet->batch + k;
                copy_cpu(gnet->outputs, gnet->output + k*gnet->outputs, 1, gen.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gen);
        float aloss = train_network(anet, merge);

        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gen.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen", 1);
            show_image(im2, "train", 1);
            save_image(im, "gen");
            save_image(im2, "train");
        }

        update_network(gnet);

        free_data(merge);
        free_data(train);
        free_data(gen);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;

        printf("%d: adv: %f | adv_avg: %f, %f rate, %lf seconds, %d images\n", i, aloss, aloss_avg, get_current_rate(gnet), sec(clock()-time), i*imgs);
        if(i%10000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(gnet, buff);
}

void train_dcgan(char *cfg, char *weight, char* modules, char *acfg, char *aweight, char* amodules, int clear, int display, char *train_images, int maxbatch)
{
    char *backup_directory = "/home/pjreddie/backup/";
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network *gnet = load_network(cfg, weight, modules, clear, 0);
    network *anet = load_network(acfg, aweight, amodules, clear, 0);

    int i, j, k;
    layer imlayer = {0};
    for (i = 0; i < gnet->n; ++i) {
        if (gnet->layers[i].out_c == 3) {
            imlayer = gnet->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", gnet->learning_rate, gnet->momentum, gnet->decay);
    int imgs = gnet->batch*gnet->subdivisions;
    i = *gnet->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(anet);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;
    args.threads=16;
    args.classes = 1;
    char *ls[2] = {"imagenet", "zzzzzzzz"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    gnet->train = 1;
    anet->train = 1;

    int x_size = gnet->inputs*gnet->batch;
    int y_size = gnet->truths*gnet->batch;
    float *imerror = cuda_make_array(0, y_size);

    float aloss_avg = -1;

    if (maxbatch == 0) maxbatch = gnet->max_batches;
    while (get_current_batch(gnet) < maxbatch) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gen = copy_data(train);
        for (j = 0; j < imgs; ++j) {
            train.y.vals[j][0] = 1;
            gen.y.vals[j][0] = 0;
        }
        time=clock();

        for(j = 0; j < gnet->subdivisions; ++j){
            get_next_batch(train, gnet->batch, j*gnet->batch, gnet->truth, 0);
            int z;
            for(z = 0; z < x_size; ++z){
                gnet->input[z] = rand_normal();
            }
            for(z = 0; z < gnet->batch; ++z){
                float mag = mag_array(gnet->input + z*gnet->inputs, gnet->inputs);
                scale_array(gnet->input + z*gnet->inputs, gnet->inputs, 1./mag);
            }

            *gnet->seen += gnet->batch;
            forward_network(gnet);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            fill_cpu(anet->truths*anet->batch, 1, anet->truth, 1);
            copy_cpu(anet->inputs*anet->batch, imlayer.output, 1, anet->input, 1);
            anet->delta_gpu = imerror;
            forward_network(anet);
            backward_network(anet);

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
            scal_gpu(imlayer.outputs*imlayer.batch, 0, gnet->layers[gnet->n-1].delta_gpu, 1);

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);

            backward_network(gnet);

            for(k = 0; k < gnet->batch; ++k){
                int index = j*gnet->batch + k;
                copy_cpu(gnet->outputs, gnet->output + k*gnet->outputs, 1, gen.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gen);

        float aloss = train_network(anet, merge);

        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gen.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen", 1);
            show_image(im2, "train", 1);
            save_image(im, "gen");
            save_image(im2, "train");
        }

        update_network(gnet);

        free_data(merge);
        free_data(train);
        free_data(gen);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;

        printf("%d: adv: %f | adv_avg: %f, %f rate, %lf seconds, %d images\n", i, aloss, aloss_avg, get_current_rate(gnet), sec(clock()-time), i*imgs);
        if(i%10000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(gnet, buff);
}

void train_colorizer(char *cfg, char *weight, char* modules, char *acfg, char *aweight, char* amodules, int clear, int display)
{
    char *train_images = "/home/pjreddie/data/imagenet/imagenet1k.train.list";
    char *backup_directory = "/home/pjreddie/backup/";
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network *net = load_network(cfg, weight, modules, clear, 0);
    network *anet = load_network(acfg, aweight, amodules, clear, 0);

    int i, j, k;
    layer imlayer = {0};
    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = net->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);

    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(net);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[2] = {"imagenet"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    int x_size = net->inputs*net->batch;

    net->delta = 0;
    net->train = 1;
    float *pixs = calloc(x_size, sizeof(float));
    float *graypixs = calloc(x_size, sizeof(float));

    anet->delta = 0;
    anet->train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs*imlayer.batch);

    float aloss_avg = -1;
    float gloss_avg = -1;

    while (get_current_batch(net) < net->max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gray = copy_data(train);
        for(j = 0; j < imgs; ++j){
            image gim = float_to_image(net->w, net->h, net->c, gray.X.vals[j]);
            grayscale_image_3c(gim);
            train.y.vals[j][0] = .95;
            gray.y.vals[j][0] = .05;
        }
        time=clock();
        float gloss = 0;

        for(j = 0; j < net->subdivisions; ++j){
            get_next_batch(train, net->batch, j*net->batch, pixs, 0);
            get_next_batch(gray, net->batch, j*net->batch, graypixs, 0);
            cuda_push_array(net->input_gpu, graypixs, net->inputs*net->batch);
            cuda_push_array(net->truth_gpu, pixs, net->truths*net->batch);

            *net->seen += net->batch;
            forward_network(net);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            copy_gpu(anet->inputs*anet->batch, imlayer.output_gpu, 1, anet->input_gpu, 1);
            fill_gpu(anet->inputs*anet->batch, .95, anet->truth_gpu, 1);
            anet->delta_gpu = imerror;
            forward_network(anet);
            backward_network(anet);

            scal_gpu(imlayer.outputs*imlayer.batch, 1./100., net->layers[net->n-1].delta_gpu, 1);

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);

            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs*imlayer.batch));

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, net->layers[net->n-1].delta_gpu, 1);

            backward_network(net);


            gloss += *net->cost /(net->subdivisions*net->batch);

            for(k = 0; k < net->batch; ++k){
                int index = j*net->batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, gray.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gray);

        float aloss = train_network(anet, merge);

        update_network(net);

        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gray.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen", 1);
            show_image(im2, "train", 1);
        }
        free_data(merge);
        free_data(train);
        free_data(gray);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;
        gloss_avg = gloss_avg*.9 + gloss*.1;

        printf("%d: gen: %f, adv: %f | gen_avg: %f, adv_avg: %f, %f rate, %lf seconds, %d images\n", i, gloss, aloss, gloss_avg, aloss_avg, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void test_lsd(char *cfg, char *weights, char* modules, char *filename, int gray)
{
    network *net = load_network(cfg, weights, modules, 0, 1);

    clock_t time;
    char buff[256];
    char *input = buff;
    int i, imlayer = 0;

    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = i;
            printf("%d\n", i);
            break;
        }
    }

    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        if(gray) grayscale_image_3c(crop);

        float *X = crop.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);

        constrain_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");
        show_image(out, "out", 1);
        show_image(crop, "crop", 0);

        free_image(im);
        free_image(resized);
        free_image(crop);
        if (filename) break;
    }
}


void run_lsd(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    int batches = find_int_arg(argc, argv, "-b", 0);
    char *file = find_char_arg(argc, argv, "-file", "/home/pjreddie/data/imagenet/imagenet1k.train.list");

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    char *acfg = argv[5];
    char *aweights = (argc > 6) ? argv[6] : 0;
    char* modules = find_char_arg(argc, argv, "-m", 0);
    char* amodules = find_char_arg(argc, argv, "-am", 0);

    if(0==strcmp(argv[2], "traingan")) train_dcgan(cfg, weights, modules, acfg, aweights, amodules, clear, display, file, batches);
    else if(0==strcmp(argv[2], "trainprog")) train_prog(cfg, weights, modules, acfg, aweights, amodules, clear, display, file, batches);
    else if(0==strcmp(argv[2], "traincolor")) train_colorizer(cfg, weights, modules, acfg, aweights, amodules, clear, display);
    else if(0==strcmp(argv[2], "gan")) test_dcgan(cfg, weights, modules);
    else if(0==strcmp(argv[2], "inter")) inter_dcgan(cfg, weights, modules);
    else if(0==strcmp(argv[2], "test")) test_lsd(cfg, weights, modules, filename, 0);
    else if(0==strcmp(argv[2], "color")) test_lsd(cfg, weights, modules, filename, 1);
}
