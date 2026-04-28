import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { Video, ResizeMode } from "expo-av";

const API_BASE_URL = "http://192.168.1.216:8000";

const hiddenKeys = [
  "wrist_range",
  "elbow_range",
  "bar_depth",
  "max_elbow",
  "min_knee",
  "max_torso",
  "min_hip",
  "bar_severity",
];

const formatLabel = (v) =>
  v
    ? String(v)
        .replaceAll("_", " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : "N/A";

const getStatusColor = (status) => {
  if (status === "poor" || status === "incomplete" || status === "drifting") {
    return "#ef4444";
  }

  if (
    status === "borderline" ||
    status === "needs_work" ||
    status === "possible" ||
    status === "shallow" ||
    status === "limited_range" ||
    status === "possibly_shallow"
  ) {
    return "#f59e0b";
  }

  return "#22c55e";
};

const getPhaseConfig = (exerciseLabel) => {
  const label = String(exerciseLabel || "").toLowerCase();

  if (label.includes("squat")) {
    return {
      text: "Setup → Descent → Bottom → Ascent → Lockout",
      highlight: "bottom",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom ⭐"],
        ["ascent", "Ascent"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("push press")) {
    return {
      text: "Setup → Dip → Drive → Lockout",
      highlight: "lockout",
      items: [
        ["setup", "Setup"],
        ["dip", "Dip"],
        ["drive", "Drive"],
        ["lockout", "Lockout ⭐"],
      ],
    };
  }

  if (label.includes("bench")) {
    return {
      text: "Setup → Descent → Bottom → Press → Lockout",
      highlight: "lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["press", "Press"],
        ["lockout", "Lockout ⭐"],
      ],
    };
  }

  return {
    text: "Setup → Pull → Mid → Finish → Lockout",
    highlight: "lockout",
    items: [
      ["setup", "Setup"],
      ["pull", "Pull"],
      ["mid", "Mid"],
      ["finish", "Finish"],
      ["lockout", "Lockout ⭐"],
    ],
  };
};

const getBestRep = (reps) => {
  if (!reps || reps.length === 0) return null;

  return reps.reduce((best, rep) => {
    return Number(rep.score || 0) > Number(best.score || 0) ? rep : best;
  }, reps[0]);
};

const getCoachingImagePath = (result) => {
  const images = result?.phase_images || {};

  return (
    images.setup ||
    images.start ||
    images.descent ||
    images.pull ||
    images.mid ||
    images.bottom ||
    images.lockout ||
    null
  );
};

const getInteractiveZones = (result) => {
  const label = String(result?.exercise_label || "").toLowerCase();
  const reps = result?.rep_feedback || [];
  const bestRep = getBestRep(reps);
  const breakdown = bestRep?.breakdown || {};

  if (label.includes("bench")) {
    return [
      {
        id: "wrists",
        title: "Wrists",
        status: breakdown.wrists || breakdown.lockout || "good",
        note:
          breakdown.lockout === "incomplete"
            ? "Finish with stacked wrists and fully extended arms."
            : "Keep wrists stacked over elbows and avoid letting them bend back.",
      },
      {
        id: "elbows",
        title: "Elbows",
        status: breakdown.elbows || "good",
        note:
          breakdown.elbows === "poor" ||
          breakdown.elbows === "severe_flare" ||
          breakdown.elbows === "borderline"
            ? "Keep elbows controlled. Avoid flaring too aggressively."
            : "Elbow path looks controlled through the press.",
      },
      {
        id: "torso",
        title: "Chest / Torso",
        status: breakdown.arch || "good",
        note: "Keep the chest stable and maintain a controlled upper-back position.",
      },
      {
        id: "bar",
        title: "Bar Path",
        status: breakdown.bar_path || breakdown.depth || "good",
        note:
          breakdown.depth === "limited_range" ||
          breakdown.depth === "possibly_shallow"
            ? "Use a full, controlled bar path from chest to lockout."
            : "Bar path looks controlled.",
      },
      {
        id: "lockout",
        title: "Lockout",
        status: breakdown.lockout || "good",
        note:
          breakdown.lockout === "incomplete"
            ? "Fully extend your arms at the top of the rep."
            : "Lockout looks solid.",
      },
    ];
  }

  if (label.includes("push press")) {
    return [
      {
        id: "dip",
        title: "Dip",
        status: breakdown.dip || "good",
        note:
          breakdown.dip === "shallow"
            ? "Use a stronger vertical dip before driving the bar overhead."
            : "Dip timing looks usable.",
      },
      {
        id: "knees",
        title: "Knees",
        status: breakdown.knees || "good",
        note:
          breakdown.knees === "borderline" || breakdown.knees === "poor"
            ? "Keep knees tracking over toes during the dip and drive."
            : "Knee tracking looks controlled.",
      },
      {
        id: "bar",
        title: "Bar Path",
        status: breakdown.bar_path || "good",
        note:
          breakdown.bar_path === "drifting"
            ? "Keep the bar close and drive straight overhead."
            : "Bar path looks controlled.",
      },
      {
        id: "lockout",
        title: "Overhead Lockout",
        status: breakdown.lockout || "good",
        note:
          breakdown.lockout === "incomplete"
            ? "Finish with elbows fully extended overhead."
            : "Overhead lockout looks solid.",
      },
    ];
  }

  if (label.includes("squat")) {
    return [
      {
        id: "neck",
        title: "Neck / Head",
        status: breakdown.neck || "good",
        note:
          breakdown.neck === "poor" || breakdown.neck === "borderline"
            ? "Keep your head aligned with your torso."
            : "Keep a neutral head position.",
      },
      {
        id: "torso",
        title: "Torso",
        status: breakdown.torso || "good",
        note:
          breakdown.torso === "poor" || breakdown.torso === "borderline"
            ? "Keep your chest taller and avoid folding forward."
            : "Torso angle looks controlled.",
      },
      {
        id: "hips",
        title: "Depth",
        status: breakdown.depth || "good",
        note:
          breakdown.depth === "poor" || breakdown.depth === "borderline"
            ? "Sink a little deeper while keeping your chest up."
            : "Depth looks good.",
      },
      {
        id: "knees",
        title: "Knees",
        status: breakdown.knees || "good",
        note:
          breakdown.knees === "poor" || breakdown.knees === "borderline"
            ? "Drive knees out and keep them tracking over toes."
            : "Knee tracking looks controlled.",
      },
      {
        id: "heels",
        title: "Heels",
        status: breakdown.heels || "good",
        note:
          breakdown.heels === "poor" || breakdown.heels === "borderline"
            ? "Keep pressure through your heels and midfoot."
            : "Heel pressure looks controlled.",
      },
    ];
  }

  return [
    {
      id: "back",
      title: "Back Position",
      status: breakdown.back || "good",
      note:
        breakdown.back === "poor" || breakdown.back === "fair"
          ? "Keep your back neutral and brace before pulling."
          : "Back position looks controlled.",
    },
    {
      id: "hips",
      title: "Hip Hinge",
      status: breakdown.hinge || "good",
      note:
        breakdown.hinge === "poor"
          ? "Push your hips back more before starting the pull."
          : "Hip hinge looks controlled.",
    },
    {
      id: "knees",
      title: "Knees",
      status: breakdown.knees || "good",
      note: "Keep knees controlled as the bar passes the legs.",
    },
    {
      id: "bar",
      title: "Bar Path",
      status: breakdown.bar_path || "good",
      note:
        breakdown.bar_path === "poor" || breakdown.bar_path === "drifting"
          ? "Keep the bar close to your body during the pull."
          : "Bar path looks controlled.",
    },
    {
      id: "lockout",
      title: "Lockout",
      status: breakdown.lockout || "good",
      note:
        breakdown.lockout === "incomplete" || breakdown.lockout === "poor"
          ? "Finish tall with hips and knees fully extended."
          : "Lockout looks solid.",
    },
  ];
};

export default function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedZone, setSelectedZone] = useState(null);

  const reset = () => {
    setResult(null);
    setSelectedZone(null);
  };

  const recordWithCamera = async () => {
    reset();

    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Camera permission required");
      return;
    }

    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      videoMaxDuration: 60,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      setVideo({
        uri: a.uri,
        name: a.fileName || "camera.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const pickFromLibrary = async () => {
    reset();

    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Library permission required");
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      setVideo({
        uri: a.uri,
        name: a.fileName || "library.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const pickFromCloud = async () => {
    reset();

    const res = await DocumentPicker.getDocumentAsync({
      type: "video/*",
      copyToCacheDirectory: true,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      setVideo({
        uri: a.uri,
        name: a.name || "cloud.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const analyzeVideo = async () => {
    if (!video) {
      Alert.alert("Pick a video first");
      return;
    }

    setLoading(true);
    setResult(null);
    setSelectedZone(null);

    try {
      const formData = new FormData();

      formData.append("file", {
        uri: video.uri,
        name: video.name,
        type: video.type,
      });

      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      console.log("BACKEND RESPONSE:", data);

      if (!res.ok) {
        throw new Error(data.detail || data.message || "Analyze request failed");
      }

      setResult(data);
    } catch (err) {
      setResult({ error: true, message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const reps = result?.rep_feedback || [];

  const overallScore =
    reps.length > 0
      ? Math.round(
          reps.reduce((sum, r) => sum + Number(r.score || 0), 0) / reps.length
        )
      : null;

  const biggestFix =
    result?.set_summary?.biggest_fix ||
    result?.rep_feedback?.[0]?.feedback?.[0] ||
    "Keep building consistent reps.";

  const phaseConfig = getPhaseConfig(result?.exercise_label);
  const zones = getInteractiveZones(result);
  const activeZone = selectedZone || zones[0];

  const coachingImagePath = getCoachingImagePath(result);
  const coachingImageUrl = coachingImagePath
    ? `${API_BASE_URL}${coachingImagePath}`
    : null;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>FormCheck AI</Text>

        <View style={styles.buttons}>
          <TouchableOpacity style={styles.primary} onPress={recordWithCamera}>
            <Text style={styles.buttonText}>Record</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondary} onPress={pickFromLibrary}>
            <Text style={styles.buttonText}>Library</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondary} onPress={pickFromCloud}>
            <Text style={styles.buttonText}>Files</Text>
          </TouchableOpacity>
        </View>

        {video && (
          <View style={styles.previewCard}>
            <Text style={styles.previewLabel}>Selected Video</Text>
            <Text style={styles.previewName}>{video.name}</Text>

            <Video
              source={{ uri: video.uri }}
              style={styles.video}
              useNativeControls
              resizeMode={ResizeMode.CONTAIN}
            />
          </View>
        )}

        <TouchableOpacity
          style={[styles.analyze, loading && styles.disabled]}
          onPress={analyzeVideo}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Analyze</Text>
          )}
        </TouchableOpacity>

        {result?.error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{result.message}</Text>
          </View>
        )}

        {result?.feedback &&
          result.rep_feedback?.length === 0 &&
          !result.error && (
            <View style={styles.errorCard}>
              <Text style={styles.errorTitle}>Video Not Usable</Text>
              <Text style={styles.errorText}>
                Camera angle is too close or unclear for reliable scoring.
              </Text>
              <Text style={styles.errorText}>How to fix:</Text>
              <Text style={styles.errorText}>• Move camera farther back</Text>
              <Text style={styles.errorText}>• Record from the side</Text>
              <Text style={styles.errorText}>
                • Keep bar, chest, elbows, hips, knees, and feet visible
              </Text>
            </View>
          )}

        {result && !result.error && (
          <View>
            <Text style={styles.exercise}>
              {result.analysis_mode === "poor_video_quality"
                ? "Video Not Usable"
                : result.rep_feedback?.length === 0
                ? "Exercise Detected — Rep Analysis Incomplete"
                : result.exercise_label}
            </Text>

            {overallScore !== null && (
              <View style={styles.summaryCard}>
                <Text style={styles.summaryLabel}>Overall Score</Text>
                <Text style={styles.summaryScore}>{overallScore}/10</Text>
                <Text style={styles.biggestFix}>Biggest Fix: {biggestFix}</Text>
              </View>
            )}

            {reps.length > 0 && (
              <View style={styles.coachMapCard}>
                <Text style={styles.coachMapTitle}>Tap a Coaching Zone</Text>
                <Text style={styles.coachMapSubtitle}>
                  Select a zone below the image to see what to fix.
                </Text>

                {coachingImageUrl ? (
                  <View style={styles.coachingImageWrap}>
                    <Image
                      source={{ uri: coachingImageUrl }}
                      style={styles.coachingImage}
                      resizeMode="cover"
                    />
                  </View>
                ) : (
                  <View style={styles.noImageBox}>
                  <Text style={styles.noImageTitle}>Coaching Image Not Generated</Text>

                  <Text style={styles.noImageText}>
                    We identified the exercise, but couldn't detect clear reps in the video.
                  </Text>

                  <Text style={styles.noImageReason}>This usually means:</Text>

                  <Text style={styles.noImageBullet}>
                    • rep start / finish frames were unclear
                  </Text>

                  <Text style={styles.noImageBullet}>
                    • movement pattern was hard to segment
                  </Text>

                  <Text style={styles.noImageBullet}>
                    • pose landmarks jumped during the lift
                  </Text>

                  <Text style={styles.noImageTip}>
                    Tomorrow's retrained model should improve this.
                  </Text>
                </View>
                )}

                <View style={styles.zoneChipGrid}>
                  {zones.map((zone) => {
                    const isActive = activeZone?.id === zone.id;

                    return (
                      <TouchableOpacity
                        key={zone.id}
                        style={[
                          styles.zoneChip,
                          {
                            backgroundColor: getStatusColor(zone.status),
                          },
                          isActive && styles.zoneChipActive,
                        ]}
                        onPress={() => setSelectedZone(zone)}
                      >
                        <Text style={styles.zoneChipText}>{zone.title}</Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>

                {activeZone && (
                  <View style={styles.zoneInfoCard}>
                    <View style={styles.zoneTitleRow}>
                      <View
                        style={[
                          styles.zoneStatusDot,
                          {
                            backgroundColor: getStatusColor(activeZone.status),
                          },
                        ]}
                      />
                      <Text style={styles.zoneTitle}>{activeZone.title}</Text>
                    </View>

                    <Text style={styles.zoneStatus}>
                      Status: {formatLabel(activeZone.status)}
                    </Text>
                    <Text style={styles.zoneNote}>{activeZone.note}</Text>
                  </View>
                )}
              </View>
            )}

            {result?.overlay_video_url && (
              <View style={styles.overlayCard}>
                <Text style={styles.overlayTitle}>Coached Replay</Text>

                <View style={styles.legendRow}>
                  <View style={styles.legendBadge}>
                    <View
                      style={[styles.legendDot, { backgroundColor: "#22c55e" }]}
                    />
                    <Text style={styles.legendText}>Your Movement</Text>
                  </View>

                  <View style={styles.legendBadge}>
                    <View
                      style={[styles.legendDot, { backgroundColor: "#3b82f6" }]}
                    />
                    <Text style={styles.legendText}>Ideal Form</Text>
                  </View>
                </View>

                <Text style={styles.overlayText}>What happened: {biggestFix}</Text>

                <Video
                  source={{ uri: `${API_BASE_URL}${result.overlay_video_url}` }}
                  style={styles.overlayVideo}
                  useNativeControls
                  shouldPlay={false}
                  resizeMode={ResizeMode.CONTAIN}
                />
              </View>
            )}

            {result?.phase_images && (
              <View style={styles.phaseCard}>
                <Text style={styles.phaseTitle}>Key Positions</Text>
                <Text style={styles.phaseText}>{phaseConfig.text}</Text>

                <View style={styles.legendRow}>
                  <View style={styles.legendBadge}>
                    <View
                      style={[styles.legendDot, { backgroundColor: "#22c55e" }]}
                    />
                    <Text style={styles.legendText}>Your Movement</Text>
                  </View>

                  <View style={styles.legendBadge}>
                    <View
                      style={[styles.legendDot, { backgroundColor: "#3b82f6" }]}
                    />
                    <Text style={styles.legendText}>Ideal Form</Text>
                  </View>
                </View>

                {phaseConfig.items.map(([key, label]) => {
                  const imageUrl = result.phase_images?.[key];
                  if (!imageUrl) return null;

                  return (
                    <View
                      key={key}
                      style={[
                        styles.phaseImageCard,
                        key === phaseConfig.highlight && styles.highlightCard,
                      ]}
                    >
                      <Text style={styles.phaseImageLabel}>{label}</Text>

                      <Image
                        source={{ uri: `${API_BASE_URL}${imageUrl}` }}
                        style={styles.phaseSingleImage}
                        resizeMode="contain"
                      />
                    </View>
                  );
                })}
              </View>
            )}

            {reps.map((rep, i) => (
              <View key={i} style={styles.card}>
                <Text style={styles.rep}>
                  Rep {rep.rep} — {rep.grade}
                </Text>

                {rep.breakdown && (
                  <View style={styles.metrics}>
                    {Object.entries(rep.breakdown)
                      .filter(([key]) => !hiddenKeys.includes(key))
                      .map(([key, value]) => (
                        <Text key={key} style={styles.metricText}>
                          {formatLabel(key)}: {formatLabel(value)}
                        </Text>
                      ))}
                  </View>
                )}

                {rep.issues?.length > 0 && (
                  <>
                    <Text style={styles.section}>Issues</Text>
                    {rep.issues.map((x, j) => (
                      <Text key={j} style={styles.issue}>
                        • {x}
                      </Text>
                    ))}
                  </>
                )}

                {rep.feedback?.length > 0 && (
                  <>
                    <Text style={styles.section}>
                      {rep.issues?.length > 0 ? "What to Fix" : "Coach Note"}
                    </Text>

                    {rep.feedback.map((item, j) => (
                      <Text key={j} style={styles.coach}>
                        → {item}
                      </Text>
                    ))}
                  </>
                )}
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0b1020" },
  content: { padding: 20, paddingBottom: 60 },

  title: { color: "#fff", fontSize: 32, fontWeight: "800" },
  buttons: { gap: 10, marginVertical: 20 },

  primary: {
    backgroundColor: "#2563eb",
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
  },

  secondary: {
    backgroundColor: "#1f2937",
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
  },

  analyze: {
    backgroundColor: "#16a34a",
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 20,
  },

  disabled: { opacity: 0.6 },
  buttonText: { color: "#fff", fontWeight: "800" },

  previewCard: {
    backgroundColor: "#111827",
    padding: 12,
    borderRadius: 14,
    marginBottom: 15,
  },

  previewLabel: { color: "#9ca3af", fontSize: 12, marginBottom: 2 },
  previewName: { color: "#fff", fontWeight: "700", marginBottom: 10 },

  video: {
    width: "100%",
    height: 220,
    backgroundColor: "#000",
    borderRadius: 12,
    marginTop: 10,
  },

  coachMapCard: {
    backgroundColor: "#111827",
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: "#8b5cf6",
  },

  coachMapTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 6,
  },

  coachMapSubtitle: {
    color: "#c4b5fd",
    fontSize: 14,
    fontWeight: "700",
    marginBottom: 14,
  },

  coachingImageWrap: {
    width: "100%",
    height: 420,
    backgroundColor: "#020617",
    borderRadius: 16,
    overflow: "hidden",
    marginBottom: 14,
    position: "relative",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },

  coachingImage: {
    width: "100%",
    height: "100%",
    backgroundColor: "#000",
  },

  noImageBox: {
    height: 260,
    backgroundColor: "#020617",
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
  },

  noImageTitle: {
  color: "#ffffff",
  fontSize: 22,
  fontWeight: "900",
  marginBottom: 12,
  textAlign: "center",
},

noImageText: {
  color: "#d1d5db",
  fontSize: 16,
  lineHeight: 24,
  textAlign: "center",
  marginBottom: 16,
},

noImageReason: {
  color: "#c4b5fd",
  fontSize: 15,
  fontWeight: "800",
  marginBottom: 10,
},

noImageBullet: {
  color: "#e5e7eb",
  fontSize: 15,
  marginBottom: 6,
},

noImageTip: {
  color: "#86efac",
  fontSize: 15,
  fontWeight: "700",
  marginTop: 14,
  textAlign: "center",
},

  zoneChipGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 14,
  },

  zoneChip: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    borderWidth: 2,
    borderColor: "rgba(255,255,255,0.7)",
  },

  zoneChipActive: {
    borderColor: "#ffffff",
    transform: [{ scale: 1.04 }],
  },

  zoneChipText: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "900",
  },

  zoneInfoCard: {
    backgroundColor: "#020617",
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.14)",
  },

  zoneTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 6,
  },

  zoneStatusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },

  zoneTitle: {
    color: "#fff",
    fontSize: 19,
    fontWeight: "900",
  },

  zoneStatus: {
    color: "#c4b5fd",
    fontSize: 14,
    fontWeight: "800",
    marginBottom: 6,
  },

  zoneNote: {
    color: "#e5e7eb",
    fontSize: 15,
    lineHeight: 22,
    fontWeight: "600",
  },

  overlayCard: {
    backgroundColor: "#052e16",
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: "#22c55e",
  },

  overlayTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 10,
  },

  overlayText: {
    color: "#86efac",
    fontSize: 16,
    fontWeight: "700",
    marginBottom: 12,
  },

  overlayVideo: {
    width: "100%",
    height: 300,
    backgroundColor: "#000",
    borderRadius: 12,
  },

  legendRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 12,
    flexWrap: "wrap",
  },

  legendBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.08)",
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },

  legendText: { color: "#fff", fontSize: 14, fontWeight: "700" },

  phaseCard: {
    backgroundColor: "#111827",
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: "#3b82f6",
  },

  phaseTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 8,
  },

  phaseText: {
    color: "#bfdbfe",
    fontSize: 15,
    fontWeight: "700",
    marginBottom: 12,
  },

  phaseImageCard: {
    backgroundColor: "#020617",
    borderRadius: 14,
    padding: 10,
    marginTop: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },

  highlightCard: {
    borderColor: "#fbbf24",
    borderWidth: 2,
  },

  phaseImageLabel: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "900",
    marginBottom: 8,
  },

  phaseSingleImage: {
    width: "100%",
    height: 280,
    backgroundColor: "#000",
    borderRadius: 12,
  },

  exercise: {
    color: "#fff",
    fontSize: 28,
    fontWeight: "900",
    marginBottom: 12,
  },

  summaryCard: {
    backgroundColor: "#111827",
    padding: 16,
    borderRadius: 16,
    marginBottom: 16,
  },

  summaryLabel: {
    color: "#9ca3af",
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 4,
  },

  summaryScore: {
    color: "#fff",
    fontSize: 34,
    fontWeight: "900",
  },

  biggestFix: {
    color: "#86efac",
    fontSize: 16,
    fontWeight: "700",
    marginTop: 10,
  },

  card: {
    backgroundColor: "#1f2937",
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
  },

  rep: {
    color: "#fff",
    fontWeight: "900",
    fontSize: 20,
    marginBottom: 10,
  },

  metrics: { marginBottom: 14 },

  metricText: {
    color: "#d1d5db",
    fontSize: 16,
    marginBottom: 3,
  },

  section: {
    color: "#fff",
    fontWeight: "900",
    fontSize: 17,
    marginTop: 6,
  },

  issue: {
    color: "#fca5a5",
    fontSize: 16,
    marginTop: 3,
  },

  coach: {
    color: "#86efac",
    fontSize: 16,
    marginTop: 5,
  },

  errorCard: {
    backgroundColor: "#7f1d1d",
    padding: 14,
    borderRadius: 12,
    marginBottom: 15,
  },

  errorText: { color: "#fff", marginTop: 4 },

  errorTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 10,
  },
});