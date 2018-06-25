clear all

fid = fopen('log_noverbose_20.txt');

ws_cnt = zeros(200,15);
acc = zeros(200,4);
for l=1:200
  tline = fgetl(fid);
  data = strsplit(tline,' ');

  for i=1:15
    ws_cnt(l,i) = str2num(data{i});
  end

  if strcmp(data{16}, "True")
    acc(l,1) = 1;
  else
    acc(l,1) = 0;
  end

  if strcmp(data{17}, "True")
    acc(l,2) = 1;
  else
    acc(l,2) = 0;
  end

  if strcmp(data{18}, "True")
    acc(l,3) = 1;
  else
    acc(l,3) = 0;
  end

  if strcmp(data{19}, "True")
    acc(l,4) = 1;
  else
    acc(l,4) = 0;
  end
end  

sum(acc,1) / 200