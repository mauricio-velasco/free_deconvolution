import numpy as np

box_corners_enum = [ 'bottom_right',
             'top_right',
             'top_left',
             'bottom_left'
              ]

box_segments_enum = []
for i in range( len(box_corners_enum) ):
       current_corner = box_corners_enum[i]
       next_corner    = box_corners_enum[(i+1) % 4]
       box_segments_enum.append( (current_corner, next_corner) )

def extend_box( box ):
       extended_box = box
       span   = extended_box['top_left']-extended_box['bottom_right']
       height = np.abs( np.imag(span) )
       width  = np.abs( np.real(span) )
       extended_box['top_right']   = extended_box['top_left']     + width
       extended_box['bottom_left'] = extended_box['bottom_right'] - width
       extended_box['height'] = height
       extended_box['width']  = width
       return extended_box

def split_box( box ):
    span = box['top_left']-box['bottom_right']
    span = ( abs(np.real(span)), abs(np.imag(span)) )
    if span[0] > span[1]:
        box1 = {
            'top_left'    : box['top_left'],
            'bottom_right': box['bottom_right']-0.5*span[0],
        }
        box2 = {
            'top_left'    : box['top_left']+0.5*span[0],
            'bottom_right': box['bottom_right'],
        }
    else:
        box1 = {
            'top_left'    : box['top_left'],
            'bottom_right': box['bottom_right']+0.5*span[1]*1.0j,
        }
        box2 = {
            'top_left'    : box['top_left']-0.5*span[1]*1.0j,
            'bottom_right': box['bottom_right'],
        }
    #
    box1 = extend_box( box1 )
    box2 = extend_box( box2 )
    return box1, box2

def box_midpoint( box ):
    return 0.5*( box['top_left'] + box['bottom_right'] )