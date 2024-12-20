

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

%%% add db to existing dbase and group

f_name = 'ns.h5';
group_ = '/gr_name';
datasetName = '/gr_name/new_data';
%% random data example
data = rand(5); 

% Check if group exists (opzionale)
info = h5info(f_name);
existing_groups = {info.Groups.Name};
if any(strcmp(existing_groups, group_))
    h5write(f_name, datasetName, data);
else
    disp('Not existing Group, just create the group');
end


%% Create  new group and add dataset
% just consider that matalb cannot create group without adding data
%% you need to add data and the process of creating a group will happen contextually
f_name = 'ns.h5';
new_gr_= '/new_gr';
datasetName = '/new_gr/new_data';
data = rand(5); 
% Group Creation and dataset add are synchronous
h5write(f_name, datasetName, data);


% Scenario 2: Create new hd5 dbase and add data
h5write(filename, datasetName, data);

f_name = 'new.h5';
group_ = '/my_group'; %
datasetName = '/my_group/my_dataset';
data = magic(4); 

% Create file and add data
h5write(f_name, datasetName, data);

% Add data to another existing (or not) griup
new_DatasetName = '/other_group/new_dataset';
newData = rand(10); 
% Add  dataset and create the group "/other_group" if not existing
h5write(filename, altroDatasetName, altroData);

