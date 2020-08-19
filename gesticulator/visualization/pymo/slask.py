class EulerReorder(BaseEstimator, TransformerMixin):
    def __init__(self, new_order):
        """
        Add a 
        """
        self.new_order = new_order
        
    
    def fit(self, X, y=None):
        self.orig_skeleton = copy.deepcopy(X[0].skeleton)
        print(self.orig_skeleton)
        return self
    
    def transform(self, X, y=None):
        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=euler_df[rzp], index=new_df.index)
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            new_track = track.clone()
            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                
                euler = [[f[1]['%s_%srotation'%(joint, rot_order[0])], f[1]['%s_%srotation'%(joint, rot_order[1])], f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in r.iterrows()]
                new_euler = [euler_reorder(f, rot_order, self.new_order, True) for f in euler]
                #new_euler = euler_reorder2(np.array(euler), rot_order, self.new_order, True)
                
                # Create the corresponding columns in the new DataFrame
                new_df['%s_%srotation'%(joint, self.new_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)
    
                new_track.skeleton[joint]['order'] = self.new_order

            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=euler_df[rzp], index=new_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            new_track = track.clone()
            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                new_order = self.orig_skeleton[joint]['order']
                print("rot_order:" + str(rot_order))
                print("new_order:" + str(new_order))

                euler = [[f[1]['%s_%srotation'%(joint, rot_order[0])], f[1]['%s_%srotation'%(joint, rot_order[1])], f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in r.iterrows()]
                #new_euler = [euler_reorder(f, rot_order, new_order, True) for f in euler]
                new_euler = euler_reorder2(np.array(euler), rot_order, self.new_order, True)

                # Create the corresponding columns in the new DataFrame
                new_df['%s_%srotation'%(joint, new_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, new_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, new_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)

                new_track.skeleton[joint]['order'] = new_order
                
            new_track.values = new_df
            Q.append(new_track)
        return Q

class Offsetter(BaseEstimator, TransformerMixin):
    def __init__(self, ref_pose):
        """
        Add a 
        """
        euler_df = ref_pose.values
        offset_cols = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]
        self.offsets = euler_df[offset_cols]
        print("offsets=" + str(self.offsets))
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the offset values
            new_df = euler_df.copy()
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                
                euler = [[f[1]['%s_%srotation'%(joint, rot_order[0])], f[1]['%s_%srotation'%(joint, rot_order[1])], f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in r.iterrows()]
                offset = [[f[1]['%s_%srotation'%(joint, rot_order[0])], f[1]['%s_%srotation'%(joint, rot_order[1])], f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in self.offsets.iterrows()]
                new_euler = [offsets_inv(offset[0], f, rot_order, True) for f in euler]
                
                # Create the corresponding columns in the new DataFrame
    
                new_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)

            new_track = track.clone()
            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the offset values
            new_df = euler_df.copy()
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                
                euler = [[f[1]['%s_%srotation'%(joint, rot_order[0])], f[1]['%s_%srotation'%(joint, rot_order[1])], f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in r.iterrows()]
                offset = [[f[1]['%s_%srotation'%(joint, rot_order[0])], f[1]['%s_%srotation'%(joint, rot_order[1])], f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in self.offsets.iterrows()]
                new_euler = [offsets(offset[0], f, rot_order, True) for f in euler]
    
                new_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)

            new_track = track.clone()
            new_track.values = new_df
            Q.append(new_track)

        return Q
