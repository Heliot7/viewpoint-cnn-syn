function rec = VOCreadrecxml(path)

x=VOCreadxml(path);
x=x.annotation;

rec.folder=x.folder;
rec.filename=x.filename;
rec.source.database=x.source.database;
if(isfield(x.source,'annotation'))
    rec.source.annotation=x.source.annotation;
end
if(isfield(x.source,'image'))
    rec.source.image=x.source.image;
end
rec.size.width=str2double(x.size.width);
rec.size.height=str2double(x.size.height);
rec.size.depth=str2double(x.size.depth);

rec.segmented=strcmp(x.segmented,'1');

rec.imgname=[x.folder '/JPEGImages/' x.filename];
rec.imgsize=str2double({x.size.width x.size.height x.size.depth});
rec.database=rec.source.database;

for i=1:length(x.object)
    rec.objects(i)=xmlobjtopas(x.object(i));
end

function p = xmlobjtopas(o)

p.class=o.name;

if isfield(o,'pose')
    if strcmp(o.pose,'Unspecified')
        p.view='';
    else
        p.view=o.pose;
    end
else
    p.view='';
end

if isfield(o,'truncated')
    p.truncated=strcmp(o.truncated,'1');
else
    p.truncated=false;
end

if isfield(o,'occluded')
    p.occluded=strcmp(o.occluded,'1');
else
    p.occluded=false;
end

if isfield(o,'difficult')
    p.difficult=strcmp(o.difficult,'1');
else
    p.difficult=false;
end

p.label=['PAS' p.class p.view];
if p.truncated
    p.label=[p.label 'Trunc'];
end
if p.occluded
    p.label=[p.label 'Occ'];
end
if p.difficult
    p.label=[p.label 'Diff'];
end

p.orglabel=p.label;

p.bbox=str2double({o.bndbox.xmin o.bndbox.ymin o.bndbox.xmax o.bndbox.ymax});

p.bndbox.xmin=str2double(o.bndbox.xmin);
p.bndbox.ymin=str2double(o.bndbox.ymin);
p.bndbox.xmax=str2double(o.bndbox.xmax);
p.bndbox.ymax=str2double(o.bndbox.ymax);

if isfield(o,'polygon')
    warning('polygon unimplemented');
    p.polygon=[];
else
    p.polygon=[];
end

if isfield(o,'mask')
    warning('mask unimplemented');
    p.mask=[];
else
    p.mask=[];
end

if isfield(o,'part')&&~isempty(o.part)
    p.hasparts=true;
    for i=1:length(o.part)
        p.part(i)=xmlobjtopas(o.part(i));
    end
else    
    p.hasparts=false;
    p.part=[];
end

if isfield(o,'point')
    p.haspoint=true;
    p.point.x=str2double(o.point.x);
    p.point.y=str2double(o.point.y);
else
    p.point=[];
end

if isfield(o,'actions')
    p.hasactions=true;
    fn=fieldnames(o.actions);
    for i=1:numel(fn)
        p.actions.(fn{i})=strcmp(o.actions.(fn{i}),'1');
    end
else
    p.hasactions=false;
    p.actions=[];
end
