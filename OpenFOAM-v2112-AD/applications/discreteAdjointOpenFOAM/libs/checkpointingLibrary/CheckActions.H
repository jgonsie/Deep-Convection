#ifndef CHECK_ACTIONS
#define CHECK_ACTIONS
#include <bitset>
namespace CHECK
{
typedef std::bitset<8> CheckFlags;
enum CHECKP_ACTIONS
{
    DO_STEP,
    DO_CALC_COST,
    DO_INTERPRET,
    TERMINATE,
    ERROR
};
} // namespace CHECK
#endif
