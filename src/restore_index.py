import io
import sys
        
def restore_index_nest_fn(ori_text,file_pre):


    fin_pre=io.StringIO(file_pre)
    #print(file_pre)
    all_pre=fin_pre.read().strip().split('\n\n')
    fin_pre.close()
    #print(len(all_pre))
    
    new_sentence=''
    restore_result=[]
    
    sentence_ori=ori_text.lower().replace('``','" ')
    sentence_ori=sentence_ori.replace("''",'" ')
    for pre_i in range(0,len(all_pre)):
        pre_lines=all_pre[pre_i].split('\n')
        #print(pre_lines)
#        print(sentence_ori)
        if len(pre_lines)>1:
            #print(pre_lines)
            sentence_pre=pre_lines[0].lower().replace('``','"')
            sentence_pre=sentence_pre.replace("''",'"')
            sentence_pre=sentence_pre.split()
            pre_result=[]
            for i in range(1,len(pre_lines)):
                pre_result.append(pre_lines[i].split('\t'))
            
            restore_sid=0
            restore_eid=0
            each_word_id=[]
            
            for i in range(0,len(sentence_pre)):

                temp_id=sentence_ori.find(sentence_pre[i])
                if temp_id<0:
                    if sentence_pre[i].find('"')>=0:
                        temp_id = sentence_ori.find(sentence_pre[i].replace('"','" '))
                    else:
                        #print('ori:',sentence_ori)
                        print('resotr index error:',sentence_pre[i])
                new_sentence+=sentence_ori[0:temp_id]
                
                restore_sid=len(new_sentence)
                restore_eid=len(new_sentence)+len(sentence_pre[i])
                each_word_id.append([str(restore_sid),str(restore_eid)])
                new_sentence+=sentence_ori[temp_id:temp_id+len(sentence_pre[i])]
                sentence_ori=sentence_ori[temp_id+len(sentence_pre[i]):]
#            print('each_word:',each_word_id)    
            for pre_ele in pre_result:
                # if len(pre_ele)>4:
                #     temp_pre_result=[each_word_id[int(pre_ele[0])][0],each_word_id[int(pre_ele[1])][1],pre_ele[3].split('|')[0],pre_ele[4]]
                # else:
                #     temp_pre_result=[each_word_id[int(pre_ele[0])][0],each_word_id[int(pre_ele[1])][1],pre_ele[3].split('|')[0],'1.00']
                temp_pre_result=[each_word_id[int(pre_ele[0])][0],each_word_id[int(pre_ele[1])][1],pre_ele[3].split('|')[0],pre_ele[4]]
                if temp_pre_result not in restore_result:
                    restore_result.append(temp_pre_result)
        else:
            sentence_pre=pre_lines[0].lower().replace('``','"')
            sentence_pre=sentence_pre.replace("''",'"')
            sentence_pre=sentence_pre.split()
           
            for i in range(0,len(sentence_pre)):

                temp_id=sentence_ori.find(sentence_pre[i])
                if temp_id<0:
                    if sentence_pre[i].find('"')>=0:
                        temp_id = sentence_ori.find(sentence_pre[i].replace('"','" '))
                    else:
                        print('resotr index error:',sentence_pre[i])
                new_sentence+=sentence_ori[0:temp_id]
                new_sentence+=sentence_ori[temp_id:temp_id+len(sentence_pre[i])]
                sentence_ori=sentence_ori[temp_id+len(sentence_pre[i]):]
#    print('resotre:',restore_result)
    return restore_result