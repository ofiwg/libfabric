#ifndef _FI_BGQ_DIRECT_TRIGGER_H_
#define _FI_BGQ_DIRECT_TRIGGER_H_

#define FABRIC_DIRECT_TRIGGER 1

#ifdef __cplusplus
extern "C" {
#endif

#ifdef FABRIC_DIRECT
/* Size must match struct fi_context */
struct fi_triggered_context {
	enum fi_trigger_event	event_type;
	union {
		struct fi_trigger_threshold	threshold;
		void				*internal[3];
	};
};
#endif

#ifdef __cplusplus
}
#endif

#endif
