

%cd('/media/zlollo/STRILA/CNR neuroscience/cebra_codes/Cebra_for_all/cebra_codes')
% db name
f_name ='manif_data.hdf5'
% gr_name
gr_name='/Cebra_behav'

% db_info
info = h5info(f_name, gr_name);

% structure of the hd5 db
h5disp(f_name)


% 
% labels = h5read(f_name, [gr_name '/labels'])';
% disp('Labels:');
% disp(labels);

%% put data in a matalb structure
%% labels
manif_db=struct()
labels_ = h5read(f_name, [gr_name '/labels']);
manif_db.labels = labels_;  

% all manifolds
for i = 2:length(info.Datasets)
    db_name = info.Datasets(i).Name;
    if startsWith(db_name, 'manif_')
        db_path = [gr_name '/' db_name];
        manif_data = h5read(f_name, db_path);
        disp(['Dataset ' db_name ':']);
        %disp(manif_data);
        manif_db.(db_name) = manif_data;


    end
end

%% extract data
labels=manif_db.labels'
data_1=manif_db.manif_20240405_212108'

plot_manif(data_1, labels)


