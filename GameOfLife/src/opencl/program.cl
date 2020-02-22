#define GLINTEROP
#define WRAP_EDGES
#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16

#ifdef GLINTEROP
__constant float4 KLEUR_AAN = (float4) (1,1,1,1), KLEUR_UIT = (float4) (0,0.27,0.37,1);
#else
__constant int    KLEUR_AAN = 0xFFFFFF, KLEUR_UIT = 0xFF006994;
#endif

uint cell_state(__global read_only uint* A, int x, int y, int w) {
	return (A[y*w + (x >> 5)] >> (x & 31)) & 1U;
}

uint cell_state2(__local read_only uint* A, int x, int y, int w) {
	return (A[y*w + (x >> 5)] >> (x & 31)) & 1U;
}

void set_cell   (__global uint* A, uint x, uint y, uint pw) { atomic_or (&(A[y*pw + (x >> 5)]),   1U << (x & 31)); }
void clear_cell (__global uint* A, uint x, uint y, uint pw) { atomic_and(&(A[y*pw + (x >> 5)]), ~(1U << (x & 31))); }

int tel_buren(__global read_only uint* A, int x, int y, int w) {
	return cell_state(A, x - 1, y - 1, w) + cell_state(A, x - 1, y    , w) + cell_state(A, x - 1, y + 1, w) +
		   cell_state(A, x    , y - 1, w) + cell_state(A, x    , y + 1, w) +
	       cell_state(A, x + 1, y - 1, w) + cell_state(A, x + 1, y    , w) + cell_state(A, x + 1, y + 1, w) ;
}

__kernel void next2
(
	__global write_only uint* volgende,
	__global read_only uint* patroon,
	int2 size_in_ints,
	int2 res,
	int2 offset,
#ifdef GLINTEROP
    write_only image2d_t screen
#else
    __global uint* screen
#endif
)
{
	int i = get_local_id(0);
	int j = get_local_id(1);
	int2 local_size = (int2)(get_local_size(0), get_local_size(1));
	int x = local_size.x * get_group_id(0) + i;
	int y = local_size.y * get_group_id(1) + j;

	__local uint lokale_cellen[LOCAL_SIZE_X * LOCAL_SIZE_Y];

	// if (x < size_in_ints.x)
	lokale_cellen[j*local_size.x + i] = cell_state(patroon, x, y, size_in_ints.x);

	barrier(CLK_LOCAL_MEM_FENCE); // Wacht tot elk item zijn cel in de lijst heeft geschreven

	if(i != 0 && i != local_size.x-1 && j > 0 && j != local_size.x-1) return;

	uint w = size_in_ints.x;

	int aantal_buren = cell_state2(lokale_cellen, x - 1, y - 1, w) + cell_state2(lokale_cellen, x - 1, y    , w) + cell_state2(lokale_cellen, x - 1, y + 1, w) +
					   cell_state2(lokale_cellen, x    , y - 1, w) + cell_state2(lokale_cellen, x    , y + 1, w) +
					   cell_state2(lokale_cellen, x + 1, y - 1, w) + cell_state2(lokale_cellen, x + 1, y    , w) + cell_state2(lokale_cellen, x + 1, y + 1, w) ;

	#ifdef GLINTEROP
	float4 kleur;
	#else
	int kleur;
	#endif

	if ((lokale_cellen[j*local_size.x+i] == 1 && aantal_buren == 2) || aantal_buren == 3) {
        set_cell(volgende, x, y, w);
        kleur = KLEUR_AAN;
    }
    else {
 		clear_cell(volgende, x, y, w); //@Notitie: volgende leegmaken na kopieren en alleen AAN pixels omdraaien is veel sneller, waarom?
        kleur = KLEUR_UIT;
    }

    x = x - 1 - offset.x;
	y = y - 1 - offset.y;
    if (x < 0 || x > res.x || y < 0 || y > res.y) return;

#ifdef GLINTEROP
	write_imagef(screen, (int2)(x, y), kleur);
#else
	screen[y*res.x + x] = kleur;
#endif
}


__kernel void next
(
	__global write_only uint* volgende,
	__global read_only uint* patroon,
	int2 size_in_ints,
	int2 res,
	int2 offset,
#ifdef GLINTEROP
    write_only image2d_t screen
#else
    __global uint* screen
#endif
)
{
	int x = (int)get_global_id(0) + 1;
	int y = (int)get_global_id(1) + 1;

	uint w = size_in_ints.x;
	if (x > w*32 || y > size_in_ints.y) return;

	uint buren = tel_buren(patroon, x, y, w);
	
	// uint i = (y+1)*w + ((x >> 5)+1);

	#ifdef GLINTEROP
	float4 kleur;
	#else
	int kleur;
	#endif

    if ((cell_state(patroon, x, y, w) == 1 && buren == 2) || buren == 3) {
        set_cell(volgende, x, y, w);
        kleur = KLEUR_AAN;
    }
    else {
 		clear_cell(volgende, x, y, w); //@Notitie: volgende leegmaken na kopieren en alleen AAN pixels omdraaien is veel sneller, waarom?
        kleur = KLEUR_UIT;
    }

    x = x - 1 - offset.x;
	y = y - 1 - offset.y;
    if (x < 0 || x > res.x || y < 0 || y > res.y) return;

#ifdef GLINTEROP
	write_imagef(screen, (int2)(x, y), kleur);
#else
	screen[y*res.x + x] = kleur;
#endif
}

__kernel void copy_cells(__global uint* A, __global uint* B, uint w, uint h) {
	uint i = get_global_id(0);
	B[i] = A[i];
	A[i] = 0;
}

// __kernel void halo_rows(__global uint* A, __global uint* B) {
// 	int id = get_global_id(0);

// 	B[id]
// }

