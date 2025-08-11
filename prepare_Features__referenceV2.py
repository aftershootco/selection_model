
## 10 Aug: 32 feature size  : Batch2 Testset
# New blur tag, old eye close, is_selected=False
#    32 features:   63.21 on AS sets,  with all set 63.65 (wherein AS is 53 \%)




def get_eye_feat(eye_batch_data):

    globalid_eye = {}
    
    gids_all = eye_batch_data["image_global_id"].values
    for gid in gids_all:
        data = eye_batch_data[eye_batch_data["image_global_id"] == gid]
        high_list = data["high_probs_eye"].values[0]

        if isinstance(high_list, str):
            high_list = ast.literal_eval(high_list)
        else:
            high_list = []
            
        # high_list = json.loads(high_list)
        # print("high_list ", high_list)

        mid_list = data["mid_probs_eye"].values[0]
        
        if isinstance(mid_list, str): # and not math.isnan(mid_list):
            mid_list = ast.literal_eval(mid_list)
        else:
            mid_list = []

        # print("high size: ", len(high_list))
        # print("mid size: ", len(mid_list))

        combined_list = []
        combined_list.extend(high_list)
        combined_list.extend(mid_list)
        # print("combined_list size: ", len(combined_list))

        
        openess_total_score_list = []
        unknown_score_list = []
        # for idx, hl in enumerate(high_list):
        for idx, hl in enumerate(combined_list):

            
            # print("hl ", hl)
            
            # hl = ast.literal_eval(hl)
            # print(hl.keys())
            
            open_prob = float(hl['open']) if hl['open'] is not None else 0
            close_prob = float(hl['closed']) if hl['closed'] is not None else 0
            partial_prob = float(hl['partial']) if hl['partial'] is not None else 0
            eye_not_visible = float(hl['eye_not_visible']) if hl['eye_not_visible'] is not None else 0
            eye_unknown = float(hl['unknown']) if hl['unknown'] is not None else 0

            # print(idx, "===> ", type(open_prob), open_prob, close_prob, partial_prob, eye_not_visible, eye_unknown)
            if (open_prob+close_prob+partial_prob) != 0.:
                openess_total_score = (open_prob + partial_prob)*open_prob + 0.5*(partial_prob/(open_prob+close_prob+partial_prob))
            else:
                openess_total_score = 0.
            unknown_score = eye_unknown + eye_not_visible
            # print("openess_total_score ", openess_total_score)
            # print("unknown_score ", unknown_score)

            openess_total_score_list.append(openess_total_score)
            unknown_score_list.append(unknown_score)
        
        
        openess_total_score_list = np.asarray(openess_total_score_list)

        if len(openess_total_score_list) > 0:
            avg_openess_score = round( np.mean(openess_total_score_list), 2)
            std_openess_score = round( np.std(openess_total_score_list), 2)
            min_openess_score = round( np.min(openess_total_score_list), 2)
            max_openess_score = round( np.max(openess_total_score_list), 2)
            median_openess_score = round( np.median(openess_total_score_list), 2)

        else:
            avg_openess_score = 0.
            std_openess_score = 0.
            min_openess_score = 0.
            max_openess_score = 0.
            median_openess_score = 0.


        # print("avg open ", avg_openess_score)

        unknown_score_list = np.asarray(unknown_score_list)

        if len(unknown_score_list) > 0:
            avg_unknown_score = round( np.mean(unknown_score_list), 2)
        else:
            avg_unknown_score = 0.
        # print("avg unknown ", avg_unknown_score)

        globalid_eye[gid] = {"openess_score_mean": avg_openess_score, "openess_score_std": std_openess_score, 
                             "openess_score_min": min_openess_score, "openess_score_max": max_openess_score, "openess_score_median":median_openess_score,
                             "unknown_score_mean": avg_unknown_score}
        
        # print(globalid_eye[gid])
        # print()
        # print('----')

    print("eye processing Done!")
    return globalid_eye






def get_blur_feat(blur_batch_data):

    globalid_blur = {}
    gids_all = blur_batch_data["image_global_id"].values

    print("Total gids_all ",len(gids_all))
    
    for itridx, gid in enumerate(gids_all):

        if itridx%5000 == 0:
            print(itridx, "/", len(gids_all))

        data = blur_batch_data[blur_batch_data["image_global_id"] == gid]
        high_list = data["high_probs_blur"].values[0]

        if isinstance(high_list, str):
            high_list = ast.literal_eval(high_list)
        else:
            high_list = []

        mid_list = data["mid_probs_blur"].values[0]
        if isinstance(mid_list, str):
            mid_list = ast.literal_eval(mid_list)
        else:
            mid_list = []

        combined_list = []
        combined_list.extend(high_list)
        # print("high size: ", len(combined_list))
        
        combined_list.extend(mid_list)
        # print("combined_list size: ", len(combined_list))

        sharpness_score_list = []
        for fidx, hb_list in enumerate(combined_list):
            sharp_prob = hb_list[0]
            soft_prob = hb_list[1]
            blur_prob = hb_list[2]

            relative_soft_blur = 0.5* soft_prob*(soft_prob + sharp_prob + blur_prob) ##  we also know: soft_prob + sharp_prob + blur_prob = 1
            sharpness_score = (1-blur_prob)*(sharp_prob) + relative_soft_blur

            sharpness_score_list.append(sharpness_score)
            
            # print(fidx, hb_list)

        sharpness_score_list = np.asarray(sharpness_score_list)

        if len(sharpness_score_list) > 0:
            sharpness_score_mean = np.mean(sharpness_score_list)
            sharpness_score_std = np.std(sharpness_score_list)
            sharpness_score_median = np.median(sharpness_score_list)
            sharpness_score_min = np.min(sharpness_score_list)
            sharpness_score_max = np.max(sharpness_score_list)

        else: 
            sharpness_score_mean = 0.
            sharpness_score_std = 0.
            sharpness_score_median = 0.
            sharpness_score_min = 0.
            sharpness_score_max = 0.

            

        if "app_blur_tag" in data.keys():
            new_blur_tag = data["app_blur_tag"].values[0]
            # print(new_blur_tag)
            globalid_blur[gid] = {"sharpness_score_mean": sharpness_score_mean, "sharpness_score_std": sharpness_score_std, 
                                  "sharpness_score_median": sharpness_score_median, "sharpness_score_min":sharpness_score_min,
                                  "sharpness_score_max": sharpness_score_max,
                                 "new_blur_tag": new_blur_tag}

        else:
            globalid_blur[gid] = {"sharpness_score_mean": sharpness_score_mean, "sharpness_score_std": sharpness_score_std, 
                                  "sharpness_score_median": sharpness_score_median, "sharpness_score_min":sharpness_score_min,
                                  "sharpness_score_max": sharpness_score_max
                                  }



        # print(globalid_blur[gid])
        # print()
    print("blur processing Done! no. of keys ", len(globalid_blur.keys()))
    return globalid_blur
    


def get_extreme_composition(bb_all_info, image_width, image_height):

    if not bb_all_info:
        return {'left_dist': image_width//2, 'right_dist': image_width//2, 'top_dist': image_height//2, 'bottom_dist': image_height//2}
    
    # Compute centers
    centers = [((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in bb_all_info]
    

    # Find extremes
    leftmost_center_x = min(centers, key=lambda c: c[0])[0]
    rightmost_center_x = max(centers, key=lambda c: c[0])[0]
    topmost_center_y = min(centers, key=lambda c: c[1])[1]
    bottommost_center_y = max(centers, key=lambda c: c[1])[1]
        
    
    # print("leftmost_center_x ", leftmost_center_x, type(leftmost_center_x))

    ## Normalize distances
    left_dist = leftmost_center_x/image_width
    right_dist = (image_width - rightmost_center_x)/image_width
    top_dist = topmost_center_y/image_height
    bottom_dist = (image_height - bottommost_center_y)/image_height


    # Calculate distances
    # print("left_dist ", left_dist)
    # print("right_dist ", right_dist)
    # print("top_dist ", top_dist)
    # print("bottom_dist ", bottom_dist)

    extreme_compos_result = {
            'left_dist': left_dist,
            'right_dist': right_dist,
            'top_dist': top_dist,
            'bottom_dist': bottom_dist }

    return extreme_compos_result


def create_features_v2_BlurEyeComposi(data_list, globalid_to_eyeFeat, globalid_to_blurFeat):

    ignored_gids_cnt = 0
    img_cnt = 0
    
    Features_X = []
    Features_Target = []
    Features_imgpaths = []
    feat_cnt = 0

    min_val = 0
    max_val = 10
    

    total_dup = len(data_list)

    valid_dup_cnt = 0

    VALID_DUP_SETS = []

    # total_dup = 1
    for idx in range(total_dup):
        dup_sets = data_list[idx]

        if idx % 1000 == 0:
            print(idx, '/', total_dup)

        ignore_dup = False

        Features_X_internal = []
        Features_Target_internal = []
        Features_imgpaths_internal = []

        for img_data in dup_sets:
            # print(img_data.keys())

            img_cnt += 1

            ## Image Level Scores
            img_blur_score = img_data['image_level']['image_blur_score']
            kiss_score = img_data['image_level']['kiss_score']
            # print(img_blur_score, kiss_score)

            global_id = img_data['global_id']

            # print("img_data ", img_data.keys())

            image_height = img_data['buffer_height'].values[0] ## exiff rotation factored in
            image_width = img_data['buffer_width'].values[0]


            # To-DO: DO NOT continue
            if global_id not in globalid_to_eyeFeat.keys():
                # print('missing ', global_id)
                ignored_gids_cnt += 1
                # continue   # i.e. ignoring face-images
                sharpness_score_mean = 0. ## Eye/face list is empty wrt. High/mid face
                sharpness_score_std = 0.
                sharpness_score_min = 0.
                sharpness_score_max = 0.
                sharpness_score_median = 0.

                openess_score_mean = 0.
                openess_score_std = 0.
                openess_score_min = 0.
                openess_score_max = 0.
                openess_score_median = 0.

                unknown_score_mean = 0.

            else:
                ## 
                eyes_feat = globalid_to_eyeFeat[global_id]
                openess_score_mean = eyes_feat['openess_score_mean']
                openess_score_std = eyes_feat['openess_score_std']
                openess_score_min = eyes_feat['openess_score_min']
                openess_score_max = eyes_feat['openess_score_max']
                openess_score_median = eyes_feat['openess_score_median']


                unknown_score_mean = eyes_feat['unknown_score_mean']
                # print(global_id, "openess_score: mean, std ", openess_score_mean, openess_score_std, " unknown: ", unknown_score_mean)

                if globalid_to_blurFeat is not None:
                    blur_feat = globalid_to_blurFeat[global_id]
                    sharpness_score_mean = blur_feat['sharpness_score_mean']
                    sharpness_score_std = blur_feat['sharpness_score_std']
                    sharpness_score_min = blur_feat['sharpness_score_min']
                    sharpness_score_max = blur_feat['sharpness_score_max']                    
                    sharpness_score_median = blur_feat['sharpness_score_median']


                ## Ignroing Face-image
                # continue

            # print(img_data['face_level'])

            blur_scoring_list = []
            closed_eyes_list = []
            emotion_list = []
            eyegaze_list = []
            blur_classification_list = []
            pose_score_list = []
            dist_centre_x_list = []
            dist_centre_y_list = []

            bb_all_info = []



            for face_info_curr in img_data['face_level']:

                # print("--> face_info_curr ", face_info_curr.keys())
                priority = face_info_curr['priority']
                bb = face_info_curr['bounding_box']

                # print(bb)
                x1 = bb['x']
                y1 = bb['y']
                x2 = x1 + bb['width']
                y2 = y1 + bb['height']

                

                bbox_centre = ( int((x1 + x2)/2), int((y1+y2)/2))

                ## Normalized distances
                dist_centre_x = abs(image_width/2 - bbox_centre[0])/(image_width/2)
                dist_centre_y = abs(image_height/2 - bbox_centre[1])/(image_height/2)


                blur_scoring = face_info_curr['blur_scoring']
                blur_class = face_info_curr['blur_class']
                blur_classification_raw = face_info_curr['blur_classification']

                # print("blur_classification_raw ", blur_classification_raw, type(blur_classification_raw) )
                ## encode blur classification
                blur_thr = 5

                if blur_classification_raw == '':
                    blur_classification_raw = 0
                    # print('********')
                
                if blur_classification_raw < blur_thr:
                    blur_classification = 0
                else:
                    blur_classification = 1
                    
                    
                closed_eyes = face_info_curr['closed_eyes']
                closed_eye_class = face_info_curr['closed_eye_class']
                
                eyegaze = face_info_curr['eyegaze']
                emotion = face_info_curr['emotion']
                pose_score = face_info_curr['pose_score']
                pose_ce = face_info_curr['pose_ce']
                pose_fs = face_info_curr['pose_fs']
                face_score = face_info_curr['face_score']

                if priority.lower() == "high" or priority.lower() == "mid":
                    blur_scoring_list.append(blur_scoring)
                    closed_eyes_list.append(closed_eyes)
                    emotion_list.append(emotion)
                    eyegaze_list.append(eyegaze)
                    blur_classification_list.append(blur_classification)
                    pose_score_list.append(pose_score)

                    dist_centre_x_list.append(dist_centre_x)
                    dist_centre_y_list.append(dist_centre_y)

                    bb_all_info.append( [(x1, y1), (x1,x2)] )


            count_faces = len(blur_scoring_list)
            if count_faces > 0:
                avg_blur_score = np.mean(blur_scoring_list)
                median_blur_score = np.median(blur_scoring_list)
                max_blur_score = np.max(blur_scoring_list)
                min_blur_score = np.min(blur_scoring_list)
                std_blur_score = np.std(blur_scoring_list)

                avg_closed_eyes = np.mean(closed_eyes_list)
                median_closed_eyes = np.median(closed_eyes_list)
                max_closed_eyes = np.max(closed_eyes_list)
                min_closed_eyes = np.min(closed_eyes_list)
                std_closed_eyes = np.std(closed_eyes_list)

                avg_emotion_list = np.mean(emotion_list)
                median_emotion_list = np.median(emotion_list)
                max_emotion_list = np.max(emotion_list)
                min_emotion_list = np.min(emotion_list)
                std_emotion_list = np.std(emotion_list)

                avg_eyegaze = np.mean(eyegaze_list)
                max_eyegaze = np.max(eyegaze_list)
                min_eyegaze = np.min(eyegaze_list)
                median_eyegaze = np.median(eyegaze_list)
                std_eyegaze = np.std(eyegaze_list)


                avg_blur_classification = np.mean(blur_classification_list)
                median_blur_classification = np.median(blur_classification_list)
                max_blur_classification = np.max(blur_classification_list)
                min_blur_classification = np.min(blur_classification_list)
                std_blur_classification = np.std(blur_classification_list)
                
                avg_pose_score = np.mean(pose_score_list)
                median_pose_score = np.median(pose_score_list)
                max_pose_score = np.max(pose_score_list)
                min_pose_score = np.min(pose_score_list)
                std_pose_score = np.std(pose_score_list)

                avg_dist_centre_x = np.mean(dist_centre_x_list)
                avg_dist_centre_y = np.mean(dist_centre_y_list)


            # print('-->', face_info_curr.keys())

            ## Image Level Target
            user_selection_target = img_data['user_selection_target']

            # ------------------------------------------

            # Defining Features 

            # feat_size = 28
            feat_size = 32 # 63.2
            # feat_size = 39

            feat_vector = np.zeros(feat_size)

            f_idx = 0
            feat_vector[f_idx] = img_blur_score; f_idx+=1
            feat_vector[f_idx] = kiss_score; f_idx+=1
            # print("f_idx ", f_idx)

            if count_faces > 0:                
                # feat_vector[2] = count_faces  
                
                MAX_FACE_COUNT = 5
                feat_vector[f_idx: f_idx + MAX_FACE_COUNT] = get_multihot(count_faces, arr_size=MAX_FACE_COUNT) ## Max 5 faces
                f_idx = f_idx + MAX_FACE_COUNT 
                # print("after face f_idx ", f_idx)


                blur_scale = 10.
                feat_vector[f_idx] = sharpness_score_mean*blur_scale ; f_idx+= 1
                feat_vector[f_idx] = sharpness_score_std*blur_scale ; f_idx+= 1
                feat_vector[f_idx] = sharpness_score_min*blur_scale ; f_idx+= 1
                feat_vector[f_idx] = sharpness_score_max*blur_scale ; f_idx+= 1
                feat_vector[f_idx] = sharpness_score_median*blur_scale ; f_idx+= 1
                

                eye_scale = 10
                feat_vector[f_idx] = min_max_bound(openess_score_mean*eye_scale, min_val, max_val); f_idx+= 1  ##New eye model
                feat_vector[f_idx] = min_max_bound(openess_score_std*eye_scale, min_val, max_val); f_idx+= 1  ##New eye model
                feat_vector[f_idx] = min_max_bound(openess_score_min*eye_scale, min_val, max_val); f_idx+= 1  ##New eye model
                feat_vector[f_idx] = min_max_bound(openess_score_max*eye_scale, min_val, max_val); f_idx+= 1  ##New eye model
                feat_vector[f_idx] = min_max_bound(openess_score_median*eye_scale, min_val, max_val); f_idx+= 1  ##New eye model

                feat_vector[f_idx] = min_max_bound(unknown_score_mean*eye_scale, min_val, max_val); f_idx+= 1  ##New eye model
                
                # '''
                feat_vector[f_idx] = avg_emotion_list; f_idx+= 1
                feat_vector[f_idx] = median_emotion_list; f_idx+= 1
                feat_vector[f_idx] = max_emotion_list; f_idx+= 1
                feat_vector[f_idx] = min_emotion_list; f_idx+= 1
                feat_vector[f_idx] = std_emotion_list; f_idx+= 1
                # '''

                '''
                emotion_hist = get_histogram_normalized(emotion_list)
                feat_vector[f_idx: f_idx + len(emotion_hist)] = emotion_hist; f_idx = f_idx + len(emotion_hist) # + 1
                '''

                # print(" emo end f_idx ", f_idx)

                feat_vector[f_idx] = avg_eyegaze; f_idx+= 1
                # feat_vector[f_idx] = std_eyegaze; f_idx+= 1
                # feat_vector[f_idx] = min_eyegaze; f_idx+= 1
                feat_vector[f_idx] = max_eyegaze; f_idx+= 1
                # feat_vector[f_idx] = median_eyegaze; f_idx+= 1

                # print(" eyegaze f_idx ", f_idx)

                '''
                # print("blur class f_idx ", f_idx)
                feat_vector[f_idx] = avg_blur_classification; f_idx+= 1
                # feat_vector[f_idx] = median_blur_classification; f_idx+= 1
                feat_vector[f_idx] = max_blur_classification; f_idx+= 1
                # feat_vector[f_idx] = min_blur_classification; f_idx+= 1
                # feat_vector[f_idx] = std_blur_classification; f_idx+= 1
                '''

                # if avg_blur_classification < 5:
                #     blur_face_cls = 0
                # else:
                #     blur_face_cls = 1
                

                feat_vector[f_idx] = avg_pose_score; f_idx+= 1
                # feat_vector[f_idx] = median_pose_score; f_idx+= 1
                # feat_vector[f_idx] = max_pose_score; f_idx+= 1
                # feat_vector[f_idx] = min_pose_score; f_idx+= 1
                # feat_vector[f_idx] = std_pose_score; f_idx+= 1

                ##Composition: Distance between BBbox centre distance & Image Centre Distance 
                # compos_scale = 5. # 10.
                compos_scale = 1.

                feat_vector[f_idx] = int(avg_dist_centre_x*compos_scale); f_idx+= 1
                feat_vector[f_idx] = int(avg_dist_centre_y*compos_scale); f_idx+= 1
                # print("f_idx ", f_idx)
                # print()

                # print("bb_all_info ", type(bb_all_info), type(image_width), type(image_height), image_width, image_height)
                extreme_compos_result = get_extreme_composition(bb_all_info, image_width, image_height)
                # print("extreme_compos_result ", extreme_compos_result)

                # print("extreme_compos_result['left_dist'] ", extreme_compos_result['left_dist'], type(extreme_compos_result['left_dist']))

                feat_vector[f_idx] = int(compos_scale * extreme_compos_result['left_dist']); f_idx+= 1
                feat_vector[f_idx] = int(compos_scale * extreme_compos_result['right_dist']); f_idx+= 1
                feat_vector[f_idx] = int(compos_scale * extreme_compos_result['top_dist']); f_idx+= 1
                feat_vector[f_idx] = int(compos_scale * extreme_compos_result['bottom_dist']); f_idx+= 1


                # ignore_dup = True
            else:
                # ignore_dup = True
                do_nothing=1



            feat_vector = np.round(feat_vector, 1)

            # ------------------------------------------

            if user_selection_target:
                target_vector = 1.
            else:
                target_vector = 0.

            # global_id = img_data['global_id']
            
            # if not ignore_dup:
                # print("ignoring in prepareFeat.. ")
            Features_X_internal.append(feat_vector)
            Features_Target_internal.append(target_vector)
            Features_imgpaths_internal.append(global_id)

        if not ignore_dup:
            Features_X.extend(Features_X_internal)
            Features_Target.extend(Features_Target_internal)
            Features_imgpaths.extend(Features_imgpaths_internal)

            valid_dup_cnt += 1

            VALID_DUP_SETS.append(data_list[idx])

    print("f_idx --> ", f_idx, "Feature size= ", feat_size)
    
    Features_X = np.asarray(Features_X)
    Features_Target = np.asarray(Features_Target)
    Features_imgpaths = np.asarray(Features_imgpaths)

    print("Features_X ", Features_X.shape)
    print("ignored_gids_cnt ", ignored_gids_cnt)
    print("img_cnt ", img_cnt)
    print("---> valid_dup_cnt in prepareFeat", valid_dup_cnt)

    return VALID_DUP_SETS, Features_X, Features_Target, Features_imgpaths



